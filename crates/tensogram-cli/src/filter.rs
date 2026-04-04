use tensogram_core::GlobalMetadata;

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

/// Look up a dot-notation key in global metadata, returning the FIRST matching value.
///
/// Supports 1-, 2-, and 3-level keys:
///   - `version` → struct field, then `common`, then `extra`
///   - `mars.param` → `common["mars"]["param"]`, then `payload[i]`, then `extra`
///   - `grib.geography.Ni` → `common["grib"]["geography"]["Ni"]`, then `payload[i]`
///
/// Search order: `common` → `payload[i]` (first match) → `extra`.
pub fn lookup_key(metadata: &GlobalMetadata, key: &str) -> Option<String> {
    let parts: Vec<&str> = key.split('.').collect();

    match parts.len() {
        1 => {
            let k = parts[0];
            if k == "version" {
                return Some(metadata.version.to_string());
            }
            if let Some(val) = metadata.common.get(k) {
                return Some(cbor_value_to_string(val));
            }
            if let Some(val) = metadata.extra.get(k) {
                return Some(cbor_value_to_string(val));
            }
        }
        2 => {
            let (ns, field) = (parts[0], parts[1]);

            if let Some(val) = lookup_in_cbor_map(metadata.common.get(ns), field) {
                return Some(val);
            }
            for entry in &metadata.payload {
                if let Some(val) = lookup_in_cbor_map(entry.get(ns), field) {
                    return Some(val);
                }
            }
            if let Some(val) = lookup_in_cbor_map(metadata.extra.get(ns), field) {
                return Some(val);
            }
        }
        3 => {
            // e.g. grib.geography.Ni → common["grib"] → Map["geography"] → Map["Ni"]
            let (top, mid, field) = (parts[0], parts[1], parts[2]);

            if let Some(inner) = lookup_nested_cbor_map(metadata.common.get(top), mid) {
                if let Some(val) = lookup_in_cbor_map(Some(inner), field) {
                    return Some(val);
                }
            }
            for entry in &metadata.payload {
                if let Some(inner) = lookup_nested_cbor_map(entry.get(top), mid) {
                    if let Some(val) = lookup_in_cbor_map(Some(inner), field) {
                        return Some(val);
                    }
                }
            }
            if let Some(inner) = lookup_nested_cbor_map(metadata.extra.get(top), mid) {
                if let Some(val) = lookup_in_cbor_map(Some(inner), field) {
                    return Some(val);
                }
            }
        }
        _ => {}
    }

    None
}

/// Look up a field inside a CBOR map value, returning the stringified value.
fn lookup_in_cbor_map(map_value: Option<&ciborium::Value>, field: &str) -> Option<String> {
    if let Some(ciborium::Value::Map(entries)) = map_value {
        for (k, v) in entries {
            if let ciborium::Value::Text(k_str) = k {
                if k_str == field {
                    return Some(cbor_value_to_string(v));
                }
            }
        }
    }
    None
}

/// Navigate one level into a CBOR map, returning the inner value.
fn lookup_nested_cbor_map<'a>(
    map_value: Option<&'a ciborium::Value>,
    key: &str,
) -> Option<&'a ciborium::Value> {
    if let Some(ciborium::Value::Map(entries)) = map_value {
        for (k, v) in entries {
            if let ciborium::Value::Text(k_str) = k {
                if k_str == key {
                    return Some(v);
                }
            }
        }
    }
    None
}

/// Test if global metadata matches a where-clause.
pub fn matches(metadata: &GlobalMetadata, clause: &WhereClause) -> bool {
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
        ciborium::Value::Text(s) => s.to_string(),
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
