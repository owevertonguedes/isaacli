/**
 * Checagem RAPIDA de runtime: abre a pagina headless por ~1s e captura erro de
 * JavaScript ao carregar. Nao joga, nao clica — so responde "quebra ao abrir?".
 *
 *   node checagem_rapida.js /caminho/do/jogo.html
 *
 * Imprime JSON {ok, problemas}. E o degrau barato ANTES do juiz comportamental:
 * pega `Cannot read properties of null` e afins em ~1s, sem acordar o juiz.
 */
const { chromium } = require('playwright');
const path = require('path');

const ALVO = process.argv[2];
const problemas = [];

async function main() {
  const navegador = await chromium.launch({ headless: true });
  const pagina = await navegador.newPage();

  pagina.on('pageerror', (e) => problemas.push(`erro de JavaScript ao carregar: ${e.message}`));
  pagina.on('console', (m) => {
    if (m.type() === 'error') problemas.push(`erro no console: ${m.text().slice(0, 140)}`);
  });

  await pagina.goto('file://' + path.resolve(ALVO), { waitUntil: 'load', timeout: 8000 });
  await pagina.waitForTimeout(600); // da tempo de setTimeout/DOMContentLoaded estourarem

  const textoVisivel = (await pagina.locator('body').innerText()).trim();
  if (textoVisivel.length < 3) problemas.push('a pagina abre em branco — nada visivel pro jogador');

  await navegador.close();
  console.log(JSON.stringify({ ok: problemas.length === 0, problemas }));
  process.exit(problemas.length === 0 ? 0 : 1);
}

main().catch((e) => {
  console.log(JSON.stringify({ ok: false, problemas: [`nao consegui abrir a pagina: ${e.message}`] }));
  process.exit(1);
});
