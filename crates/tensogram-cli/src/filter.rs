use tensogram_core::Metadata;

/// A parsed where-clause filter.
#[derive(Debug)]
pub struct WhereClause {
    pub key: String,
    pub op: FilterOp,
    pub values: Vec<String>,
}

#[derive(Debug, PartialEq)]
pub enum FilterOp {
    Eq,
    NotEq,
}

/// Parse a where-clause string like "mars.param=2t/10u" or "mars.class!=od".
pub fn parse_where(input: &str) -> Result<WhereClause, String> {
    if let Some((key, rest)) = input.split_once("!=") {
        let values = rest.split('/').map(|s| s.to_string()).collect();
        Ok(WhereClause {
            key: key.to_string(),
            op: FilterOp::NotEq,
            values,
        })
    } else if let Some((key, rest)) = input.split_once('=') {
        let values = rest.split('/').map(|s| s.to_string()).collect();
        Ok(WhereClause {
            key: key.to_string(),
            op: FilterOp::Eq,
            values,
        })
    } else {
        Err(format!(
            "invalid where-clause: {input} (expected key=value or key!=value)"
        ))
    }
}

/// Look up a dot-notation key in metadata, returning the string value if found.
pub fn lookup_key(metadata: &Metadata, key: &str) -> Option<String> {
    let parts: Vec<&str> = key.split('.').collect();

    // Check top-level extra keys first (namespaced like "mars.param")
    if parts.len() == 2 {
        if let Some(ciborium::Value::Map(entries)) = metadata.extra.get(parts[0]) {
            for (k, v) in entries {
                if let ciborium::Value::Text(k_str) = k {
                    if k_str == parts[1] {
                        return Some(cbor_value_to_string(v));
                    }
                }
            }
        }
        // Also check per-object extra (return first match)
        for obj in &metadata.objects {
            if let Some(ciborium::Value::Map(entries)) = obj.extra.get(parts[0]) {
                for (k, v) in entries {
                    if let ciborium::Value::Text(k_str) = k {
                        if k_str == parts[1] {
                            return Some(cbor_value_to_string(v));
                        }
                    }
                }
            }
        }
    }

    // Check single-key top-level fields
    if parts.len() == 1 {
        match parts[0] {
            "version" => return Some(metadata.version.to_string()),
            key => {
                if let Some(val) = metadata.extra.get(key) {
                    return Some(cbor_value_to_string(val));
                }
            }
        }
    }

    None
}

/// Test if metadata matches a where-clause.
pub fn matches(metadata: &Metadata, clause: &WhereClause) -> bool {
    match lookup_key(metadata, &clause.key) {
        Some(actual) => match clause.op {
            FilterOp::Eq => clause.values.iter().any(|v| v == &actual),
            FilterOp::NotEq => clause.values.iter().all(|v| v != &actual),
        },
        None => matches!(clause.op, FilterOp::NotEq), // missing key: Eq fails, NotEq passes
    }
}

fn cbor_value_to_string(value: &ciborium::Value) -> String {
    match value {
        ciborium::Value::Text(s) => s.clone(),
        ciborium::Value::Integer(i) => {
            let n: i128 = (*i).into();
            n.to_string()
        }
        ciborium::Value::Float(f) => f.to_string(),
        ciborium::Value::Bool(b) => b.to_string(),
        ciborium::Value::Null => "null".to_string(),
        _ => format!("{value:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_eq() {
        let clause = parse_where("mars.param=2t/10u").unwrap();
        assert_eq!(clause.key, "mars.param");
        assert_eq!(clause.op, FilterOp::Eq);
        assert_eq!(clause.values, vec!["2t", "10u"]);
    }

    #[test]
    fn test_parse_neq() {
        let clause = parse_where("mars.class!=od").unwrap();
        assert_eq!(clause.key, "mars.class");
        assert_eq!(clause.op, FilterOp::NotEq);
        assert_eq!(clause.values, vec!["od"]);
    }

    #[test]
    fn test_parse_invalid() {
        assert!(parse_where("no_operator").is_err());
    }
}
