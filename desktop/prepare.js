// Build step (runs on the build machine, never on the user's PC): assembles web/ — the app page,
// both models, the ONNX Runtime engine, jsPDF and the fonts — so the installed app needs no internet.
const fs = require("fs"), path = require("path");

const root = path.join(__dirname, ".."), docs = path.join(root, "docs"), out = path.join(__dirname, "web");
const mod = (p) => path.join(__dirname, "node_modules", p);
fs.rmSync(out, { recursive: true, force: true });
fs.mkdirSync(out, { recursive: true });

// models and the WebAssembly engine, as the page already loads them (base64 chunks)
for (const dir of ["m/bir6", "m/small", "ort/jsep"]) fs.cpSync(path.join(docs, dir), path.join(out, dir), { recursive: true });

// runtime scripts the page otherwise takes from CDNs
fs.mkdirSync(path.join(out, "vendor/ort"), { recursive: true });
for (const f of ["ort.webgpu.min.js", "ort-wasm-simd-threaded.jsep.mjs"]) fs.copyFileSync(mod("onnxruntime-web/dist/" + f), path.join(out, "vendor/ort", f));
fs.copyFileSync(mod("jspdf/dist/jspdf.umd.min.js"), path.join(out, "vendor/jspdf.umd.min.js"));

// fonts: Latin subsets only, named as the page names them
const fonts = [
  ["@fontsource-variable/bricolage-grotesque", "opsz.css", "'Bricolage Grotesque Variable'", "'Bricolage Grotesque'"],
  ["@fontsource/instrument-sans", "400.css"], ["@fontsource/instrument-sans", "500.css"], ["@fontsource/instrument-sans", "600.css"],
  ["@fontsource/jetbrains-mono", "400.css"], ["@fontsource/jetbrains-mono", "500.css"],
];
fs.mkdirSync(path.join(out, "fonts"), { recursive: true });
let css = "";
for (const [pkg, file, from, to] of fonts) {
  let text = fs.readFileSync(mod(`${pkg}/${file}`), "utf8");
  for (const block of text.split("\n\n")) {
    if (!/latin-(opsz|\d+)-normal|latin-\d+-normal|-latin-opsz-normal/.test(block) || /latin-ext/.test(block)) continue;
    const m = block.match(/url\(\.\/files\/([^)]+)\)/);
    if (!m) continue;
    fs.copyFileSync(mod(`${pkg}/files/${m[1]}`), path.join(out, "fonts", m[1]));
    css += (from ? block.replace(from, to) : block).replace(/, url\(\.\/files\/[^)]+\.woff\) format\('woff'\)/, "").replace("./files/", "./") + "\n\n";
  }
}
fs.writeFileSync(path.join(out, "fonts", "fonts.css"), css);

// the page itself, pointed at the local copies
const ORIGIN = "app://rubylith/";
let page = fs.readFileSync(path.join(docs, "index.html"), "utf8");
const swaps = [
  [/<link rel="preconnect" href="https:\/\/fonts\.googleapis\.com">\n?/, ""],
  [/<link rel="stylesheet" href="https:\/\/fonts\.googleapis\.com\/css2[^"]*">/, `<link rel="stylesheet" href="${ORIGIN}fonts/fonts.css">`],
  ['"https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/"', `"${ORIGIN}vendor/ort/"`],
  ['"https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"', `"${ORIGIN}vendor/jspdf.umd.min.js"`],
];
for (const [from, to] of swaps) {
  const before = page; page = page.replace(from, to);
  if (page === before) throw new Error("prepare.js: the page no longer contains " + from);
}
if (/https:\/\/(cdn\.jsdelivr|cdnjs|fonts\.googleapis)/.test(page)) throw new Error("prepare.js: an online dependency is left in the page");
fs.writeFileSync(path.join(out, "index.html"), page);

const size = (d) => fs.readdirSync(d, { withFileTypes: true }).reduce((n, e) => n + (e.isDirectory() ? size(path.join(d, e.name)) : fs.statSync(path.join(d, e.name)).size), 0);
console.log(`web/ ready: ${(size(out) / 1048576).toFixed(0)} MB, ${css.split("@font-face").length - 1} font faces`);
