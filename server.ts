import { join } from 'path';
import { ensureData } from './src/download';

const PUBLIC = join(import.meta.dir, 'public');
const DATA   = join(import.meta.dir, 'data');

const MIME: Record<string, string> = {
  '.html': 'text/html',
  '.js':   'application/javascript',
  '.pdbqt': 'text/plain',
  '.pdb':  'text/plain',
  '.dat':  'text/plain',
};

Bun.serve({
  port: 8080,

  async fetch(req: Request) {
    const url = new URL(req.url);
    let pathname = url.pathname === '/' ? '/index.html' : url.pathname;

    let filePath: string;
    if (pathname.startsWith('/systems/')) {
      filePath = join(DATA, pathname);
    } else if (pathname.startsWith('/data/')) {
      filePath = join(DATA, pathname.slice(5));
    } else {
      filePath = join(PUBLIC, pathname);
    }

    if (!filePath.startsWith(PUBLIC) && !filePath.startsWith(DATA)) {
      return new Response('Forbidden', { status: 403 });
    }

    const f = Bun.file(filePath);
    const exists = await f.exists();
    if (!exists) {
      return new Response('Not found', { status: 404 });
    }

    const ext = filePath.substring(filePath.lastIndexOf('.'));
    const type = MIME[ext] ?? 'application/octet-stream';

    return new Response(f, { headers: { 'Content-Type': type } });
  },

  error(err: Error) {
    console.error(err);
    return new Response('Internal Server Error', { status: 500 });
  },
});

ensureData().then(() => {
  console.log(`Open http://localhost:8080 in your browser`);
});
