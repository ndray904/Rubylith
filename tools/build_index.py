# docs/rubylith.html is the page as published to claude.ai (the artifact host adds the document skeleton).
# This wraps it into docs/index.html so the same app runs from any static web server.
import pathlib
app = pathlib.Path(__file__).resolve().parent.parent / "docs"
body = (app / "rubylith.html").read_text(encoding="utf-8")
(app / "index.html").write_text(
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    '<style>[hidden]{display:none!important}</style>\n</head>\n<body>\n' + body + '\n</body>\n</html>\n',
    encoding="utf-8")
print("wrote", app / "index.html")
