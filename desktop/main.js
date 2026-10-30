// Rubylith for Windows: the web app in its own window, served from the install folder.
// A private app:// scheme (instead of file://) gives the page a secure origin, so WebGPU,
// fetch() and the model worker behave exactly as they do on the website.
const { app, BrowserWindow, protocol, net, shell, Menu } = require("electron");
const path = require("path");
const { pathToFileURL } = require("url");

const WEB = app.isPackaged ? path.join(process.resourcesPath, "web") : path.join(__dirname, "web");

protocol.registerSchemesAsPrivileged([
  { scheme: "app", privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true, stream: true } },
]);

// the page checks for this to load the bundled models without caching a second copy
app.userAgentFallback = `${app.userAgentFallback} RubylithDesktop/${app.getVersion()}`;

function createWindow() {
  const win = new BrowserWindow({
    width: 1400, height: 900, minWidth: 900, minHeight: 600,
    backgroundColor: "#141413", title: "Rubylith", autoHideMenuBar: true,
    webPreferences: { contextIsolation: true, sandbox: true },
  });
  // the source-code link and any other outside links open in the user's browser
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/.test(url)) shell.openExternal(url);
    return { action: "deny" };
  });
  win.webContents.on("will-navigate", (e, url) => {
    if (!url.startsWith("app://")) { e.preventDefault(); if (/^https?:/.test(url)) shell.openExternal(url); }
  });
  win.loadURL("app://rubylith/index.html");
}

app.whenReady().then(() => {
  protocol.handle("app", (req) => {
    const rel = decodeURIComponent(new URL(req.url).pathname).replace(/^\/+/, "");
    const file = path.normalize(path.join(WEB, rel || "index.html"));
    if (!file.startsWith(WEB + path.sep)) return new Response("Not found", { status: 404 });
    return net.fetch(pathToFileURL(file).toString());
  });
  Menu.setApplicationMenu(null);
  createWindow();
});

app.on("window-all-closed", () => app.quit());
