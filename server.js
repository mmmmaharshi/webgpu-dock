const http = require('http');
const fs = require('fs');
const path = require('path');

const types = {
  '.html': 'text/html',
  '.js':   'application/javascript',
  '.pdbqt': 'text/plain',
  '.dat':  'text/plain',
};

const server = http.createServer((req, res) => {
  let p = req.url === '/' ? '/index.html' : decodeURIComponent(req.url);
  const file = path.join(__dirname, p);
  // basic path traversal guard
  if (!file.startsWith(__dirname)) {
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
