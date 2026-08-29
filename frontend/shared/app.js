/* Shared frontend helpers for the Autonomic Agentic BSN dashboard.
   No build step -- loaded as an ES module directly by the browser. */

/**
 * Fetch JSON, returning null instead of throwing when it is unavailable.
 *
 * Phase result files only exist after their script has been run, so "missing"
 * is a normal state, not an error. Returning null lets callers render a
 * friendly "run the Phase N script first" message rather than dumping a raw
 * console error the user has to interpret.
 *
 * @param {string} path
 * @returns {Promise<any|null>}
 */
export async function loadJSON(path) {
  try {
    const res = await fetch(path, { cache: 'no-store' });
    if (!res.ok) {
      console.info(`[loadJSON] ${path} -> HTTP ${res.status} (not generated yet?)`);
      return null;
    }
    return await res.json();
  } catch (err) {
    console.info(`[loadJSON] ${path} unavailable: ${err.message}`);
    return null;
  }
}

/**
 * Try several paths in order, returning the first that loads.
 *
 * Needed because the documented serve command is `cd frontend &&
 * python -m http.server 8000`, which cannot reach ../results/ (http.server
 * refuses paths above its root). The phase scripts therefore mirror their
 * output into frontend/results/, and we prefer that copy, falling back to the
 * canonical ../results/ path when the server is rooted at the project instead.
 *
 * @param {string[]} paths
 * @returns {Promise<any|null>}
 */
export async function loadJSONAny(paths) {
  for (const p of paths) {
    const data = await loadJSON(p);
    if (data !== null) return data;
  }
  return null;
}

/** Escape text for safe innerHTML interpolation. */
export function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

/** Render the standard "not yet generated" panel. */
export function notGenerated(el, scriptName, files) {
  el.innerHTML = `
    <div class="notice">
      <strong>Not yet generated.</strong>
      Run the Phase 1 script first:
      <div style="margin:.6em 0"><code>python ${esc(scriptName)}</code></div>
      Expected output: ${files.map((f) => `<code>${esc(f)}</code>`).join(' or ')}
    </div>`;
}

/** Format a number as a fixed-precision string, or a dash when absent. */
export function fmt(v, digits = 3) {
  return (v === null || v === undefined || Number.isNaN(v))
    ? '&ndash;' : Number(v).toFixed(digits);
}
