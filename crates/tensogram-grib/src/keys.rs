/// MARS namespace keys to extract from each GRIB message.
///
/// These cover identification, spatial, temporal, and parameter dimensions.
/// Keys that are not present in a particular GRIB message are silently skipped.
pub const MARS_KEYS: &[&str] = &[
    // Identification
    "class",
    "type",
    "stream",
    "expver",
    // Parameter
    "param",
    "shortName",
    "name",
    "paramId",
    "discipline",
    "parameterCategory",
    "parameterNumber",
    // Vertical
    "level",
    "typeOfLevel",
    "levtype",
    // Temporal
    "date",
    "dataDate",
    "time",
    "dataTime",
    "stepRange",
    "step",
    "stepUnits",
    // Spatial / grid
    "gridType",
    "Ni",
    "Nj",
    "numberOfPoints",
    "latitudeOfFirstGridPointInDegrees",
    "longitudeOfFirstGridPointInDegrees",
    "latitudeOfLastGridPointInDegrees",
    "longitudeOfLastGridPointInDegrees",
    "iDirectionIncrementInDegrees",
    "jDirectionIncrementInDegrees",
    // Packaging
    "bitsPerValue",
    "packingType",
    // Origin
    "centre",
    "subCentre",
    "generatingProcessIdentifier",
];
