use std::collections::hash_map::Entry;
use std::collections::BTreeMap;
use std::fmt;
use std::time;

pub struct Timing {
    sections: BTreeMap<String, SectionTimer>,
}

struct SectionTimer {
    start: time::Instant,
    duration: time::Duration,
}

impl Timing {
    pub fn new() -> Self {
        Timing {
            sections: BTreeMap::new(),
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

    pub fn elapsed(&self, name: &str) -> time::Duration {
        match self.sections.get(name) {
            Some(timer) => timer.duration,
            None => time::Duration::from_millis(0),
        }
    }

    pub fn dump(&self) {
        println!("{}", self);
    }

    pub fn dump_divided(&self, divisor: usize) {
        println!("=== Timing / {} ===", divisor);
        for (name, timer) in &self.sections {
            println!(
                "{}: {}ns",
                name,
                duration_as_total_nanos(&timer.duration) / divisor as u64
            );
        }
    }
}

pub fn duration_as_total_nanos(duration: &time::Duration) -> u64 {
    duration.as_secs() * 1_000_000_000 + duration.subsec_nanos() as u64
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
