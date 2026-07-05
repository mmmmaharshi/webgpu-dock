import http, { IncomingMessage, ServerResponse } from 'http';
import fs from 'fs';
import path from 'path';

const PUBLIC = path.join(__dirname, 'public');
const DATA   = path.join(__dirname, '..', 'data');

const types: Record<string, string> = {
  '.html': 'text/html',
  '.js':   'application/javascript',
  '.pdbqt': 'text/plain',
  '.pdb':  'text/plain',
  '.dat':  'text/plain',
};

const server = http.createServer((req: IncomingMessage, res: ServerResponse) => {
  let url = req.url === '/' ? '/index.html' : decodeURIComponent(req.url ?? '');

  let file: string;
  if (url.startsWith('/systems/')) {
    file = path.join(DATA, url);
  } else if (url.startsWith('/data/')) {
    file = path.join(DATA, url.slice(5));
  } else {
    file = path.join(PUBLIC, url);
  }

  if (!file.startsWith(PUBLIC) && !file.startsWith(DATA)) {
    res.writeHead(403); res.end('Forbidden'); return;
  }

  fs.readFile(file, (err: NodeJS.ErrnoException | null, data: Buffer) => {
    if (err) { res.writeHead(404); res.end('Not found'); return; }
    res.writeHead(200, { 'Content-Type': types[path.extname(file)] || 'text/plain' });
    res.end(data);
  });
});

function tryListen(port: number): void {
  server.listen(port, () => {
    console.log(`Open http://localhost:${port} in your browser`);
  });
  server.on('error', (e: NodeJS.ErrnoException) => {
    if (e.code === 'EADDRINUSE') {
      console.log(`Port ${port} in use, trying ${port + 1}`);
      tryListen(port + 1);
    } else {
      throw e;
    }
  });
}

tryListen(8080);
