export interface UnitGroup {
  units: string[];
  toDisplay: (v: number, displayUnit: string) => number;
  toNative: (v: number, displayUnit: string) => number;
}

const TEMPERATURE: UnitGroup = {
  units: ['K', '°C'],
  toDisplay: (v, u) => u === '°C' ? v - 273.15 : v,
  toNative:  (v, u) => u === '°C' ? v + 273.15 : v,
};

const PRESSURE: UnitGroup = {
  units: ['Pa', 'hPa'],
  toDisplay: (v, u) => u === 'hPa' ? v / 100 : v,
  toNative:  (v, u) => u === 'hPa' ? v * 100 : v,
};

const GROUPS: UnitGroup[] = [TEMPERATURE, PRESSURE];

/** Returns the unit group whose native unit (first entry) matches, or undefined. */
export function getUnitGroup(nativeUnit: string): UnitGroup | undefined {
  return GROUPS.find(g => g.units[0] === nativeUnit);
}
