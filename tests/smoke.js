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
  console.log('week cards:', await p.$$eval('#cards .card', c => c.length), '| with a posted line:', await p.$$eval('#cards .card .mkt', c => c.filter(x => x.textContent.startsWith('Posted')).length));
  for (const s of ['total','etotal','espread','shoot','kick','gap']) { await p.click(`[data-sort="${s}"]`); await p.waitForTimeout(120); }
  await p.click('nav [data-view="shootout"]'); await p.waitForTimeout(300);
  console.log('shootout rows:', await p.$$eval('#sotable tbody tr', r => r.length));
  const top = await p.$$eval('#sotable tbody tr', rs => rs.slice(0,6).map(r => [...r.querySelectorAll('td')].map(t => t.innerText.split('\n')[0]).filter(Boolean).join(' | ')));
  console.log('--- shootout, top 6 ---'); top.forEach(t => console.log('  ' + t));
  await p.click('[data-ssort="kick"]'); await p.waitForTimeout(200);
  await p.click('nav [data-view="lines"]'); await p.waitForTimeout(300);
  console.log('line-move rows:', await p.$$eval('#lmtable tbody tr', r => r.length));
  const lm = await p.$$eval('#lmtable tbody tr', rs => rs.slice(0,5).map(r => [...r.querySelectorAll('td')].map(t => t.innerText.split('\n')[0]).filter(Boolean).join(' | ')));
  console.log('--- biggest total moves, top 5 ---'); lm.forEach(t => console.log('  ' + t));
  for (const s of ['smove','mopen','kick']) { await p.click(`[data-lsort="${s}"]`); await p.waitForTimeout(120); }
  await p.click('#lm-hide'); await p.waitForTimeout(120);
  await p.click('nav [data-view="rank"]'); await p.waitForTimeout(200);
  console.log('ranking rows:', await p.$$eval('#ranktable tbody tr', r => r.length));
  await p.click('nav [data-view="bt"]'); await p.waitForTimeout(300);
  console.log('backtest sections:', await p.$$eval('#view-bt h2', h => h.length));
  console.log(errs.length ? 'ERRORS:\n' + errs.join('\n') : 'no errors');
  if (errs.length) process.exitCode = 1;
  await b.close();
})();
