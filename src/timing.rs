use std::time;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt;

pub struct Timing {
    sections: HashMap<String, SectionTimer>,
}

struct SectionTimer {
    start: time::Instant,
    duration: time::Duration,
}

impl Timing {
    pub fn new() -> Self {
        Timing {
            sections: HashMap::new(),
        }
    }

    pub fn start(&mut self, name: &str) {
        let now = time::Instant::now();
        let entry = self.sections
            .entry(name.to_string())
            .or_insert_with(|| SectionTimer {
                start: now,
                duration: time::Duration::from_millis(0),
            });

        entry.start = now;
    }

    pub fn stop(&mut self, name: &str) {
        match self.sections.get_mut(name) {
            Some(timer) => {
                let now = time::Instant::now();
                timer.duration += now - timer.start;
            }
            None => {
                panic!("Undefined timer section {}", name);
            }
        }
    }

    pub fn dump(&self) {
        println!("{}", self);
    }
}

impl fmt::Display for Timing {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "=== Timing ===");
        for (name, timer) in &self.sections {
            writeln!(
                f,
                "{}: {}.{:04}s",
                name,
                timer.duration.as_secs(),
                timer.duration.subsec_nanos() / 1000000
            );
        }
        Ok(())
    }
}
