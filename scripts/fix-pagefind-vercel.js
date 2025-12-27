// scripts/fix-pagefind-vercel.js
import fs from 'fs';
import { execSync } from 'child_process';
import { join } from 'path';

// Vercel 环境检测
const isVercel = process.env.VERCEL === '1';

if (isVercel) {
  console.log('🔄 Running on Vercel, applying fixes...');
  
  // 确保 dist 目录存在
  const distPath = join(process.cwd(), 'dist');
  if (!fs.existsSync(distPath)) {
    console.error('❌ dist directory not found at:', distPath);
    process.exit(1);
  }
  
  // 运行 PageFind
  try {
    // 使用绝对路径
    execSync('npx pagefind --site dist', { 
      stdio: 'inherit',
      cwd: process.cwd()
    });
    console.log('✅ PageFind built successfully on Vercel');
  } catch (error) {
    console.error('❌ PageFind build failed:', error.message);
    // 尝试备选方案
    try {
      execSync('node node_modules/pagefind/pagefind.js --site dist', {
        stdio: 'inherit',
        cwd: process.cwd()
      });
      console.log('✅ PageFind built with alternative method');
    } catch (error2) {
      console.error('❌ All PageFind attempts failed');
    }
  }
} else {
  console.log('🏠 Running locally, skipping Vercel-specific fixes');
}