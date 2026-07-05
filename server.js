const http = require('http');
const fs = require('fs');
const path = require('path');

const PUBLIC = path.join(__dirname, 'public');
const DATA   = path.join(__dirname, 'data');

const types = {
  '.html': 'text/html',
  '.js':   'application/javascript',
  '.pdbqt': 'text/plain',
  '.pdb':  'text/plain',
  '.dat':  'text/plain',
};

const server = http.createServer((req, res) => {
  let url = req.url === '/' ? '/index.html' : decodeURIComponent(req.url);

  // Route data requests to the data/ directory
  let file;
  if (url.startsWith('/systems/')) {
    file = path.join(DATA, url);           // /systems/... → data/systems/...
  } else if (url.startsWith('/data/')) {
    file = path.join(DATA, url.slice(5));   // /data/...    → data/...
  } else {
    file = path.join(PUBLIC, url);          // everything else → public/
  }

  // path traversal guard
  if (!file.startsWith(PUBLIC) && !file.startsWith(DATA)) {
    res.writeHead(403); res.end('Forbidden'); return;
  }

  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end('Not found'); return; }
    res.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'text/plain' });
    res.end(data);
  });
});

function tryListen(port) {
  server.listen(port, () => {
    console.log(`Open http://localhost:${port} in your browser`);
  });
  server.on('error', (e) => {
    if (e.code === 'EADDRINUSE') {
      console.log(`Port ${port} in use, trying ${port + 1}`);
      tryListen(port + 1);
    } else {
      throw e;
    }
  });
}

tryListen(8080);
