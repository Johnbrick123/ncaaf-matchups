// Smoke test for the rendered board.  Usage: node tests/smoke.js docs/index.html
// Needs: npm i playwright && npx playwright install chromium
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined });
  const p = await b.newPage();
  const errs = [];
  p.on('pageerror', e => errs.push('PAGEERROR: ' + e.message));
  p.on('console', m => { if (m.type()==='error' && !/fonts\.g/.test(m.text())) errs.push('CONSOLE: ' + m.text()); });
  await p.goto('file://' + require('path').resolve(process.argv[2] || 'docs/index.html'));
  await p.waitForTimeout(800);
  console.log('subline:', await p.$eval('#subline', e => e.textContent));
  console.log('week cards:', await p.$$eval('#cards .card', c => c.length), '| price calcs:', await p.$$eval('.calc .c-out', c => c.filter(x => x.textContent.includes('clears')).length));
  for (const s of ['total','shoot','oread','bfloor','kick','gap']) { await p.click(`[data-sort="${s}"]`); await p.waitForTimeout(120); }
  await p.click('nav [data-view="shootout"]'); await p.waitForTimeout(300);
  console.log('shootout rows:', await p.$$eval('#sotable tbody tr', r => r.length), '| calibration rows:', await p.$$eval('#so-cal tbody tr', r => r.length));
  const top = await p.$$eval('#sotable tbody tr', rs => rs.slice(0,6).map(r => [...r.querySelectorAll('td')].map(t => t.innerText.split('\n')[0]).filter(Boolean).join(' | ')));
  console.log('--- shootout, top 6 ---'); top.forEach(t => console.log('  ' + t));
  await p.click('[data-ssort="edge"]'); await p.waitForTimeout(200);
  const ed = await p.$$eval('#sotable tbody tr', rs => rs.slice(0,6).map(r => [...r.querySelectorAll('td')].map(t => t.innerText.split('\n')[0]).filter(Boolean).join(' | ')));
  console.log('--- historical over-bet edge, top 6 ---'); ed.forEach(t => console.log('  ' + t));
  await p.click('nav [data-view="rank"]'); await p.waitForTimeout(200);
  console.log('ranking rows:', await p.$$eval('#ranktable tbody tr', r => r.length));
  await p.click('nav [data-view="bt"]'); await p.waitForTimeout(300);
  console.log('backtest sections:', await p.$$eval('#view-bt h2', h => h.length));
  console.log(errs.length ? 'ERRORS:\n' + errs.join('\n') : 'no errors');
  if (errs.length) process.exitCode = 1;
  await b.close();
})();
