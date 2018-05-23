/// Dataset must be shareable between multiple threads,
/// so we can train separate parts of training data batch
/// in parallel.
/// 'static  here means that Dataset cannot have borrows with lifetimes
/// shorter than 'static. Note that Dataset itself doesn't have to live indefinitely.
pub trait Dataset: Sync + Send + 'static {
    fn slices_for_cursor(&self, current_example_index: usize) -> (&[f64], &[f64]);
    fn examples_count(&self) -> usize;
    fn input_size(&self) -> usize;
    fn label_size(&self) -> usize;
}
