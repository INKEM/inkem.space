// scripts/copy-pagefind.js
import fs from 'fs';
import path from 'path';

const src = './dist/pagefind';
const dest = './public/pagefind';

if (fs.existsSync(src)) {
  fs.rmSync(dest, { recursive: true, force: true });
  fs.cpSync(src, dest, { recursive: true });
  console.log('✅ Copied pagefind to public/');
} else {
  console.error('❌ dist/pagefind not found!');
  process.exit(1);
}