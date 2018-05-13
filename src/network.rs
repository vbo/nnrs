use std::fmt;
use std::mem;
use std::collections::HashSet;
use std::fs::File;
use serde_json;
use math::Matrix;
use math::Vector;

const LEARNING_RATE: f64 = 0.1;
pub const BATCH_SIZE: usize = 1000;

#[derive(Clone)]
pub struct Network {
    parameters: NetworkParameters,
    trainer: NetworkTrainer,
    predictor: NetworkPredictor,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NetworkParameters {
    layers: Vec<Layer>,
    input_layer: LayerID,
    output_layer: LayerID,
}

#[derive(Clone)]
pub struct NetworkTrainer {
    layers: Vec<TrainingLayer>,
}

#[derive(Clone)]
pub struct NetworkPredictor {
    layer_activations: Vec<Vector>,
}

impl NetworkParameters {
    pub fn input_layer(&self) -> LayerID {
        self.input_layer
    }
    pub fn output_layer(&self) -> LayerID {
        self.output_layer
    }
    fn get_layer_mut(&mut self, id: LayerID) -> &mut Layer {
        &mut self.layers[id.0]
    }
    fn get_layer(&self, id: LayerID) -> &Layer {
        &self.layers[id.0]
    }

    pub fn load_from_file(path: &str) -> Self {
        let file = File::open(path).unwrap();
        let mut nn: Self = serde_json::from_reader(&file).unwrap();
        return nn;
    }

    pub fn write_to_file(&self, output_path: &str) {
        let file = File::create(&output_path).unwrap();
        serde_json::to_writer(file, &self).unwrap();
    }

    fn calc_layers_order_internal(
        &self,
        scheduled_currently: &mut HashSet<LayerID>,
        layers_order: &mut Vec<LayerID>,
        current_id: LayerID,
    ) {
        scheduled_currently.insert(current_id);
        for dependency in &self.get_layer(current_id).dependencies {
            if layers_order.contains(&dependency.id) {
                continue;
            }
            if scheduled_currently.contains(&dependency.id) {
                panic!("Dependency cycle!");
            } else {
                self.calc_layers_order_internal(scheduled_currently, layers_order, dependency.id);
            }
        }
        layers_order.push(current_id);
    }

    fn calc_layers_order(&self) -> Vec<LayerID> {
        let mut scheduled_currently = HashSet::with_capacity(self.layers.len());
        let mut layers_order = Vec::with_capacity(self.layers.len());
        let output_layer = self.output_layer();
        self.calc_layers_order_internal(&mut scheduled_currently, &mut layers_order, output_layer);

        return layers_order;
    }
}

impl NetworkTrainer {
    pub fn for_parameters(parameters: &NetworkParameters) -> Self {
        let mut trainer = NetworkTrainer { layers: Vec::new() };

        for layer in &parameters.layers {
            let mut training_layer = TrainingLayer::new(layer.size);
            for dep in &layer.dependencies {
                training_layer
                    .dependencies
                    .push(TrainingLayerDependency::empty(
                        dep.weights.rows,
                        dep.weights.cols,
                    ));
            }
            trainer.layers.push(training_layer);
        }

        return trainer;
    }

    fn get_layer_mut(&mut self, id: LayerID) -> &mut TrainingLayer {
        &mut self.layers[id.0]
    }

    pub fn backward_propagation(
        &mut self,
        parameters: &NetworkParameters,
        predictor: &NetworkPredictor,
        true_outputs: &Vector,
    ) {
        // TODO(vbo): calculate order once and reuse, invalidate on adding
        // layers / dependencies
        let mut layers_order = parameters.calc_layers_order();
        layers_order.reverse();

        for layer in &mut self.layers {
            layer.error.fill_with(0.0);
        }

        // Compute output error.
        let output_layer_id = parameters.output_layer();
        {
            let output_layer = self.get_layer_mut(output_layer_id);
            let output_layer_activations = predictor.get_layer_activations(output_layer_id);
            assert!(
                true_outputs.rows == output_layer_activations.rows,
                "Labels size mismatch with output layer: {} != {}",
                true_outputs.rows,
                output_layer_activations.rows
            );

            true_outputs.sub(&output_layer_activations, &mut output_layer.error);
        }

        for layer_id in layers_order {
            match parameters.get_layer(layer_id).kind {
                LayerKind::Input => continue,
                _ => self.compute_layer_backward(parameters, predictor, layer_id),
            }
        }
    }

    /// Weights adjustment from info accumulated during backward propagation calls.
    pub fn apply_batch(&mut self, parameters: &mut NetworkParameters) {
        for (layer, training_layer) in parameters.layers.iter_mut().zip(self.layers.iter_mut()) {
            layer.bias += avg_by_batch(training_layer.bias_batch_pd);
            training_layer.bias_batch_pd = 0.0;

            let joined_dependencies = layer
                .dependencies
                .iter_mut()
                .zip(training_layer.dependencies.iter_mut());

            for (dependency, training_dependency) in joined_dependencies {
                training_dependency.weights_batch_pd.apply(&avg_by_batch);
                dependency
                    .weights
                    .add(&training_dependency.weights_batch_pd);
                training_dependency.weights_batch_pd.fill_with(0.0);
            }
        }
    }

    fn calc_grad_prefix(activations: &Vector, error: &Vector, res: &mut Vector) {
        assert!(
            activations.rows == error.rows && error.rows == res.rows,
            "Invalid dimentions for output error grad prefix PD: {}x1, {}x1, {}x1",
            activations.rows,
            error.rows,
            res.rows
        );

        for row in 0..activations.rows {
            res.mem[row] =
                2.0 * error.mem[row] * activations.mem[row] * (1.0 - activations.mem[row]);
        }
    }

    fn calc_weights_pd_and_add(
        error_grad_prefix: &Vector,
        previous_activations: &Vector,
        res: &mut Matrix,
    ) {
        assert!(
            error_grad_prefix.rows == res.rows && res.cols == previous_activations.rows,
            "Invalid dimentions for output weights PD: {}x1, {}x1, {}x{}",
            error_grad_prefix.rows,
            previous_activations.rows,
            res.rows,
            res.cols
        );

        for row in 0..res.rows {
            let row_start = row * res.cols;
            for col in 0..res.cols {
                res.mem[row_start + col] +=
                    error_grad_prefix.mem[row] * previous_activations.mem[col];
            }
        }
    }

    fn get_layer_and_deps_mut(
        &mut self,
        parameters: &NetworkParameters,
        layer_id: LayerID,
    ) -> (&mut TrainingLayer, Vec<&mut TrainingLayer>) {
        let layer = parameters.get_layer(layer_id);
        let layers_buf: *mut TrainingLayer = self.layers.as_mut_ptr();
        let cur_layer: *mut TrainingLayer = unsafe { layers_buf.offset(layer_id.0 as isize) };
        let mut deps_layers_vec = Vec::with_capacity(layer.dependencies.len());
        for dep in &layer.dependencies {
            let dep_layer: *mut TrainingLayer = unsafe { layers_buf.offset(dep.id.0 as isize) };
            deps_layers_vec.push(unsafe { &mut *dep_layer });
        }

        return (unsafe { &mut *cur_layer }, deps_layers_vec);
    }

    fn compute_layer_backward(
        &mut self,
        parameters: &NetworkParameters,
        predictor: &NetworkPredictor,
        layer_id: LayerID,
    ) {
        let layer = parameters.get_layer(layer_id);
        let (training_layer, mut dep_layers) = self.get_layer_and_deps_mut(parameters, layer_id);
        let mut grad_prefix = Vector::new(parameters.get_layer(layer_id).size).init_with(0.0);
        NetworkTrainer::calc_grad_prefix(
            &predictor.get_layer_activations(layer_id),
            &training_layer.error,
            &mut grad_prefix,
        );
        training_layer.bias_batch_pd += grad_prefix.calc_sum();
        for (dep_index, training_dependency) in training_layer.dependencies.iter_mut().enumerate() {
            let dependency = &layer.dependencies[dep_index];
            NetworkTrainer::calc_weights_pd_and_add(
                &grad_prefix,
                &predictor.get_layer_activations(dependency.id),
                &mut training_dependency.weights_batch_pd,
            );
            // TODO(lenny): fuse together transpose and add_dot_vec
            let mut weights_t =
                Matrix::new(dependency.weights.cols, dependency.weights.rows).init_with(0.0);
            dependency.weights.transpose(&mut weights_t);
            weights_t.add_dot_vec(&grad_prefix, &mut dep_layers[dep_index].error);
        }
    }
}

impl NetworkPredictor {
    pub fn for_parameters(parameters: &NetworkParameters) -> Self {
        let mut activations = Vec::new();

        for layer in &parameters.layers {
            activations.push(Vector::new(layer.size).init_with(0.0));
        }

        NetworkPredictor {
            layer_activations: activations,
        }
    }

    pub fn get_layer_activations(&self, id: LayerID) -> &Vector {
        &self.layer_activations[id.0]
    }

    pub fn get_layer_activations_mut(&mut self, id: LayerID) -> &mut Vector {
        &mut self.layer_activations[id.0]
    }

    pub fn predict(&mut self, parameters: &NetworkParameters, input_data: &[f64]) -> &Vector {
        let layers_order = parameters.calc_layers_order();

        {
            let input_id = parameters.input_layer();
            let input_layer_activations = self.get_layer_activations_mut(input_id);
            assert!(
                input_data.len() == input_layer_activations.rows,
                "Invalid input dimensions."
            );
            input_layer_activations.copy_from_slice(input_data);
        }

        for layer_id in layers_order {
            match parameters.get_layer(layer_id).kind {
                LayerKind::Input => continue,
                _ => self.compute_layer_forward(parameters, layer_id),
            }
        }

        return self.get_layer_activations(parameters.output_layer());
    }

    fn get_layer_activations_mut_and_deps(
        &mut self,
        parameters: &NetworkParameters,
        layer_id: LayerID,
    ) -> (&mut Vector, Vec<&Vector>) {
        let layer = parameters.get_layer(layer_id);
        let layers_buf: *mut Vector = self.layer_activations.as_mut_ptr();
        let cur_layer: *mut Vector = unsafe { layers_buf.offset(layer_id.0 as isize) };
        let mut deps_layers_vec = Vec::with_capacity(layer.dependencies.len());
        for dep in &layer.dependencies {
            let dep_layer: *mut Vector = unsafe { layers_buf.offset(dep.id.0 as isize) };
            deps_layers_vec.push(unsafe { &*dep_layer });
        }

        return (unsafe { &mut *cur_layer }, deps_layers_vec);
    }

    fn compute_layer_forward(&mut self, parameters: &NetworkParameters, layer_id: LayerID) {
        let layer = parameters.get_layer(layer_id);
        let layer_bias = layer.bias;
        let (activations, dep_layer_activations) =
            self.get_layer_activations_mut_and_deps(parameters, layer_id);
        activations.fill_with(0.0);
        for (dep_index, dependency) in layer.dependencies.iter().enumerate() {
            let dep_activations = &dep_layer_activations[dep_index];
            let weights = &dependency.weights;
            weights.add_dot_vec(dep_activations, activations);
        }
        activations.apply(|x| x + layer_bias);
        activations.apply(sigmoid);
    }
}

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct LayerID(usize);
impl fmt::Display for LayerID {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LayerID:{}", self.0)
    }
}

#[derive(Serialize, Deserialize, Clone)]
enum LayerKind {
    Input,
    Output,
    Hidden,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / ((-x).exp() + 1.0)
}

fn avg_by_batch(x: f64) -> f64 {
    x * LEARNING_RATE / BATCH_SIZE as f64
}

impl Network {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Network::from_parameters(NetworkParameters {
            layers: {
                let mut v = Vec::new();
                v.push(Layer::new(LayerKind::Input, input_size));
                v.push(Layer::new(LayerKind::Output, output_size));
                v
            },
            input_layer: LayerID(0),
            output_layer: LayerID(1),
        })
    }

    pub fn from_parameters(parameters: NetworkParameters) -> Self {
        Network {
            trainer: NetworkTrainer::for_parameters(&parameters),
            predictor: NetworkPredictor::for_parameters(&parameters),
            parameters: parameters,
        }
    }

    pub fn input_layer(&self) -> LayerID {
        self.parameters.input_layer()
    }
    pub fn output_layer(&self) -> LayerID {
        self.parameters.output_layer()
    }

    /*
    fn get_layer_mut(&mut self, id: LayerID) -> &mut Layer {
        &mut self.layers[id.0]
    }
    fn get_layer_activations_mut(&mut self, id: LayerID) -> &mut Vector {
        &mut self.layers[id.0].activations
    }
    fn get_layer(&self, id: LayerID) -> &Layer {
        &self.layers[id.0]
    }

    /// Layers vector is parallel to layer.dependencies.
    fn get_layer_and_deps_mut(&mut self, layer_id: LayerID) -> (&mut Layer, Vec<&mut Layer>) {
        // NOTE(vbo): maybe separate Nodes and Weights to different vectors.
        let layers_buf: *mut Layer = self.layers.as_mut_ptr();
        let cur_layer: *mut Layer = unsafe { layers_buf.offset(layer_id.0 as isize) };
        let mut deps_layers_vec = Vec::with_capacity(self.get_layer(layer_id).dependencies.len());
        for dep in &mut self.get_layer_mut(layer_id).dependencies {
            let dep_layer: *mut Layer = unsafe { layers_buf.offset(dep.id.0 as isize) };
            deps_layers_vec.push(unsafe { &mut *dep_layer });
        }

        return (unsafe { &mut *cur_layer }, deps_layers_vec);
    }
    */

    fn calc_layers_order(&self) -> Vec<LayerID> {
        self.parameters.calc_layers_order()
    }

    pub fn predict(&mut self, input_data: &[f64]) -> &Vector {
        self.predictor.predict(&self.parameters, input_data)
    }

    pub fn backward_propagation(&mut self, true_outputs: &Vector) {
        self.trainer
            .backward_propagation(&self.parameters, &self.predictor, true_outputs);
    }

    pub fn add_hidden_layer(&mut self, rows: usize) -> LayerID {
        self.parameters
            .layers
            .push(Layer::new(LayerKind::Hidden, rows));
        self.trainer.layers.push(TrainingLayer::new(rows));
        self.predictor
            .layer_activations
            .push(Vector::new(rows).init_with(0.0));
        return LayerID(self.parameters.layers.len() - 1);
    }

    pub fn add_layer_dependency(&mut self, source_id: LayerID, target_id: LayerID) {
        assert!(
            source_id.0 != target_id.0,
            "Self dependency is not allowed for {}",
            source_id
        );
        assert!(
            source_id.0 < self.parameters.layers.len(),
            "Invalid dependency source"
        );
        assert!(
            target_id.0 < self.parameters.layers.len(),
            "Invalid dependency target"
        );

        let mut dup_found = false;
        for dep in &self.parameters.layers[source_id.0].dependencies {
            if dep.id == target_id {
                dup_found = true;
            }
        }
        assert!(!dup_found, "Duplicate dependency");

        let source_size = self.parameters.get_layer(source_id).size;
        let target_size = self.parameters.get_layer(target_id).size;
        // TODO(parallel): Matrices can be shared between threads readonly, while
        // activations we need to copy.
        let weights = Matrix::new(source_size, target_size).init_rand();
        let layer_dependency = LayerDependency {
            id: target_id,
            weights: weights,
        };
        self.parameters.layers[source_id.0]
            .dependencies
            .push(layer_dependency);

        let training_layer_dependency = TrainingLayerDependency::empty(source_size, target_size);
        self.trainer.layers[source_id.0]
            .dependencies
            .push(training_layer_dependency);
    }

    /// Weights adjustment from info accumulated during backward propagation calls.
    /// Will also reset this info, preparing for a new batch.
    pub fn apply_batch(&mut self) {
        self.trainer.apply_batch(&mut self.parameters);
    }

    pub fn write_to_file(&self, output_path: &str) {
        self.parameters.write_to_file(output_path);
    }

    pub fn load_from_file(path: &str) -> Self {
        Network::from_parameters(NetworkParameters::load_from_file(path))
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct Layer {
    kind: LayerKind,
    dependencies: Vec<LayerDependency>,
    bias: f64,
    size: usize,
}

impl Layer {
    fn new(kind: LayerKind, rows: usize) -> Self {
        Layer {
            kind: kind,
            dependencies: Vec::new(),
            bias: 0.0,
            size: rows,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct LayerDependency {
    id: LayerID,
    weights: Matrix,
}

#[derive(Clone)]
struct TrainingLayer {
    dependencies: Vec<TrainingLayerDependency>,
    error: Vector,
    bias_batch_pd: f64,
}

impl TrainingLayer {
    fn new(rows: usize) -> Self {
        TrainingLayer {
            dependencies: Vec::new(),
            error: Vector::new(rows).init_with(0.0),
            bias_batch_pd: 0.0,
        }
    }
}

#[derive(Clone)]
struct TrainingLayerDependency {
    weights_batch_pd: Matrix,
}

impl TrainingLayerDependency {
    pub fn empty(source_size: usize, target_size: usize) -> Self {
        TrainingLayerDependency {
            weights_batch_pd: Matrix::new(source_size, target_size).init_with(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_order_forward() {
        let mut nn = Network::new(5, 2);
        let inputs_id = nn.input_layer();
        let l1_id = nn.add_hidden_layer(2);
        let l2_id = nn.add_hidden_layer(2);
        let outputs_id = nn.output_layer();
        nn.add_layer_dependency(outputs_id, l2_id);
        nn.add_layer_dependency(l2_id, l1_id);
        nn.add_layer_dependency(l1_id, inputs_id);

        let layer_order = nn.calc_layers_order();
        assert_eq!(layer_order, vec![inputs_id, l1_id, l2_id, outputs_id])
    }

    #[test]
    fn layer_order_skip() {
        let mut nn = Network::new(5, 2);
        let inputs_id = nn.input_layer();
        let l1_id = nn.add_hidden_layer(2);
        let l2_id = nn.add_hidden_layer(2);
        let outputs_id = nn.output_layer();
        nn.add_layer_dependency(outputs_id, l2_id);
        nn.add_layer_dependency(outputs_id, l1_id);
        nn.add_layer_dependency(l2_id, l1_id);
        nn.add_layer_dependency(l2_id, inputs_id);
        nn.add_layer_dependency(l1_id, inputs_id);

        let layer_order = nn.calc_layers_order();
        assert_eq!(layer_order, vec![inputs_id, l1_id, l2_id, outputs_id])
    }

    #[test]
    #[should_panic(expected = "Dependency cycle")]
    fn layer_order_cycle() {
        let mut nn = Network::new(5, 2);
        let inputs_id = nn.input_layer();
        let outputs_id = nn.output_layer();
        let l1_id = nn.add_hidden_layer(2);
        nn.add_layer_dependency(outputs_id, l1_id);
        nn.add_layer_dependency(l1_id, inputs_id);
        nn.add_layer_dependency(inputs_id, outputs_id);

        let layer_order = nn.calc_layers_order();
    }
}
