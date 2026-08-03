/**
 * Juiz COMPORTAMENTAL: abre o jogo num navegador de verdade, clica, digita,
 * e verifica se a pagina REAGE. Sintaxe valida nao conta como jogo.
 *
 *   node juiz_comportamental.js /caminho/do/jogo.html
 *
 * Sai com codigo 0 se passou, 1 se falhou. Imprime JSON com os problemas —
 * eles sao a materia-prima da aula do professor, entao precisam ser concretos:
 * "cliquei em #btn e nada mudou" ensina; "nao funciona" nao ensina nada.
 */
const { chromium } = require('playwright');
const path = require('path');
const SPECS = require('./specs');

const ALVO = process.argv[2];
// --visivel: abre o Chrome na tela e joga devagar, pra dar pra acompanhar a olho.
const VISIVEL = process.argv.includes('--visivel');
const problemas = [];
const evidencias = [];

async function main() {
  const navegador = await chromium.launch({
    headless: !VISIVEL,
    slowMo: VISIVEL ? 350 : 0,
    args: VISIVEL ? ['--window-size=900,700'] : [],
  });
  const pagina = await navegador.newPage();
  if (VISIVEL) await pagina.waitForTimeout(400);

  // Erro de JS em tempo de execucao e falha dura: o jogo quebrou ao carregar.
  pagina.on('pageerror', (e) => problemas.push(`erro de JavaScript ao rodar: ${e.message}`));
  pagina.on('console', (m) => {
    if (m.type() === 'error') problemas.push(`erro no console: ${m.text().slice(0, 140)}`);
  });

  await pagina.goto('file://' + path.resolve(ALVO), { waitUntil: 'load' });

  // 1) A pagina mostra alguma coisa?
  const textoVisivel = (await pagina.locator('body').innerText()).trim();
  if (textoVisivel.length < 3) problemas.push('a pagina abre em branco — nada visivel pro jogador');

  // 2) Sobrou placeholder em vez de codigo?
  const html = await pagina.content();
  const marcadores = ['JavaScript aqui', 'CSS aqui', 'seu codigo aqui', 'TODO', 'codigo aqui'];
  for (const m of marcadores) {
    if (html.toLowerCase().includes(m.toLowerCase()))
      problemas.push(`deixou o placeholder "${m}" no lugar de codigo real`);
  }

  // 3) Ha algo interativo? Atributo [onclick] nao pega handler ligado via JS
  //    (el.onclick = ...), entao conta tambem elementos com cursor:pointer.
  const interativos = await pagina.locator('button, input, [onclick], canvas, select, a[href="#"]').count();
  const clicaveis = await pagina.evaluate(() =>
    [...document.querySelectorAll('body *')].filter((el) => getComputedStyle(el).cursor === 'pointer').length);
  if (interativos === 0 && clicaveis === 0)
    problemas.push('nao ha nenhum botao, input ou canvas — o jogador nao tem como jogar');

  // 4) TESTE GENERICO: interagir e exigir que o estado da pagina mude.
  //    Se existe spec pro jogo, ela substitui este passo (e a versao forte dele —
  //    o teste generico ja foi fraudado uma vez, ver calibracao/fraudes/).
  const spec = SPECS[path.basename(ALVO)];
  if (!spec) {
    const antes = await pagina.locator('body').innerText();
    let interagiu = false;

    const botoes = await pagina.locator('button, [onclick]').all();
    for (const b of botoes.slice(0, 6)) {
      try {
        const campos = await pagina.locator('input:not([type=checkbox]):not([type=radio])').all();
        for (const c of campos.slice(0, 2)) {
          const tipo = await c.getAttribute('type');
          await c.fill(tipo === 'number' ? '42' : 'a', { timeout: 2000 });
        }
        await b.click({ timeout: 2000 });
        interagiu = true;
        await pagina.waitForTimeout(250);
      } catch (e) { /* botao pode estar coberto/invisivel; tenta o proximo */ }
    }

    // Jogos de teclado: tenta algumas teclas tambem.
    if (!interagiu || (await pagina.locator('canvas').count()) > 0) {
      for (const k of ['ArrowRight', 'Space', 'a', 'Enter']) {
        await pagina.keyboard.press(k).catch(() => {});
        await pagina.waitForTimeout(120);
      }
      interagiu = true;
    }

    const depois = await pagina.locator('body').innerText();
    const mudouCanvas = await pagina.evaluate(() => {
      const c = document.querySelector('canvas');
      if (!c) return null;
      try {
        const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
        return d.some((v, i) => i % 4 !== 3 && v !== 0);  // algum pixel pintado
      } catch { return null; }
    });

    if (interagiu && antes === depois && mudouCanvas !== true) {
      problemas.push(
        'interagi com todos os botoes/inputs/teclas e o conteudo da pagina nao mudou nada — ' +
        'o jogo nao responde ao jogador'
      );
      evidencias.push(`texto antes e depois identico: ${JSON.stringify(antes.slice(0, 120))}`);
    }
  }

  // 5) SPEC POR JOGO: roteiro que so passa se o jogo existir de verdade.
  //    (O juiz generico foi fraudado uma vez — o modelo copiou a mensagem de erro
  //    pro <h1> e passou. A spec fecha isso: dica coerente, contador que sobe de
  //    1 em 1, carta que volta a esconder — nao da pra satisfazer com texto.)
  if (spec && problemas.length === 0) {
    // recarrega pra spec comecar do estado inicial, limpa de qualquer clique
    const recarregar = async () => {
      await pagina.goto('file://' + path.resolve(ALVO), { waitUntil: 'load' });
      await pagina.waitForTimeout(200);
    };
    await recarregar();
    try {
      await spec(pagina, problemas, evidencias, recarregar);
    } catch (e) {
      problemas.push(`spec do jogo falhou ao rodar: ${e.message.slice(0, 160)}`);
    }
  }

  if (VISIVEL) await pagina.waitForTimeout(1200);  // deixa ver o resultado final
  await navegador.close();

  const ok = problemas.length === 0;
  console.log(JSON.stringify({ ok, problemas, evidencias }, null, 2));
  process.exit(ok ? 0 : 1);
}

main().catch((e) => {
  console.log(JSON.stringify({ ok: false, problemas: [`o juiz nao conseguiu abrir o jogo: ${e.message}`] }, null, 2));
  process.exit(1);
});
