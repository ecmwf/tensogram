/** Displays the full metadata JSON for the currently selected field. */

import { useAppStore } from '../../store/useAppStore';

export function MetadataPanel() {
  const { fileIndex, selectedObject } = useAppStore();

  if (!fileIndex || !selectedObject) return null;

  const obj = fileIndex.variables.find(
    (v) =>
      v.msgIndex === selectedObject.msgIdx &&
      v.objIndex === selectedObject.objIdx,
  );

  if (!obj) return null;

  const json = JSON.stringify(obj.metadata, null, 2);

  return (
    <div className="metadata-panel">
      <h2>Metadata</h2>
      <pre className="metadata-pre">
        <code>{json}</code>
      </pre>
    </div>
  );
}
