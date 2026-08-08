/**
 * Roteiros de teste POR JOGO — fecham a fraude a metrica generica.
 *
 * O juiz generico ("o conteudo mudou?") foi fraudado: o Isaac copiou a mensagem
 * de erro do juiz pra dentro do <h1> e passou sem fazer o jogo. Cada spec aqui
 * so passa se o jogo existir DE VERDADE: dicas coerentes entre si, contador que
 * sobe de 1 em 1, cartas que voltam a esconder. Impossivel satisfazer com texto.
 *
 * Contrato: cada spec e async (pagina, problemas, evidencias) => void.
 * Empurra problema concreto ("chutei 50, disse maior; chutei 75, disse maior de
 * novo com secreto<75") — e material de aula, precisa ensinar.
 */

// -- helpers ------------------------------------------------------------------

/** Extrai o primeiro numero que aparece perto de uma palavra-chave no texto. */
function numeroPerto(texto, regex) {
  const m = texto.match(regex);
  return m ? parseInt(m[1], 10) : null;
}

async function corpo(pagina) {
  return (await pagina.locator('body').innerText()).trim();
}

// -- adivinha: busca binaria exige numero secreto de verdade -------------------

/**
 * Joga busca binaria. `interpretacao` resolve a ambiguidade de "maior":
 *   'secreto' = "maior" fala do numero secreto  → chutar mais alto
 *   'palpite' = "maior" fala do MEU palpite     → chutar mais baixo
 * Devolve {ok, motivo, evidencia}.
 */
async function jogarAdivinha(pagina, interpretacao) {
  const campo = pagina.locator('input[type=number], input[type=text], input:not([type])').first();
  const botao = pagina.locator('button').first();
  if ((await campo.count()) === 0 || (await botao.count()) === 0) {
    return { ok: false, motivo: 'adivinha: nao achei campo de palpite + botao — sem como jogar' };
  }

  let lo = 1, hi = 100, tentativasAntes = null;
  for (let rodada = 1; rodada <= 7; rodada++) {
    const chute = Math.floor((lo + hi) / 2);
    await campo.fill(String(chute), { timeout: 2000 });
    await botao.click({ timeout: 2000 });
    await pagina.waitForTimeout(200);
    const texto = (await corpo(pagina)).toLowerCase();

    // Contador de tentativas deve subir de 1 em 1, se existir.
    const cont = numeroPerto(texto, /tentativas?\D{0,15}?(\d+)/i);
    if (cont !== null && tentativasAntes !== null && cont !== tentativasAntes + 1) {
      return { ok: false, motivo:
        `adivinha: contador de tentativas foi de ${tentativasAntes} pra ${cont} apos 1 palpite — deveria subir exatamente 1` };
    }
    tentativasAntes = cont;

    const disseMaior = /maior|acima|mais alto/.test(texto);
    const disseMenor = /menor|abaixo|mais baixo/.test(texto);
    const acertou = /acertou|parab|venceu|correto|ganhou/.test(texto);

    if (acertou) {
      return { ok: true, evidencia: `adivinha: acertei ${chute} em ${rodada} palpites via busca binaria` };
    }
    let subir;  // o secreto esta acima do chute?
    if (disseMaior && !disseMenor) subir = interpretacao === 'secreto';
    else if (disseMenor && !disseMaior) subir = interpretacao !== 'secreto';
    else {
      return { ok: false, motivo:
        `adivinha: chutei ${chute} e a pagina nao respondeu "maior" nem "menor" nem "acertou". Texto: ${JSON.stringify(texto.slice(0, 120))}` };
    }
    if (subir) lo = chute + 1; else hi = chute - 1;
    if (lo > hi) {
      return { ok: false, contradicao: true, motivo:
        'adivinha: as dicas se contradizem — seguindo "maior/menor" o intervalo possivel ficou vazio (nao existe numero secreto coerente)' };
    }
  }
  // 7 palpites de busca binaria fecham [1,100]: um jogo real termina em "acertou"
  // ou mantem o intervalo coerente. (Com 6 rodadas, responder sempre "maior" ainda
  // era coerente com secreto=100 — fraude medida pela calibracao. 7 fecha isso.)
  return { ok: true, evidencia: `adivinha: 7 palpites com dicas coerentes, intervalo final [${lo},${hi}]` };
}

async function adivinha(pagina, problemas, evidencias, recarregar) {
  // "maior" pode falar do secreto ("o numero e maior") ou do palpite ("seu chute
  // e maior"). Tenta a 1a interpretacao; se der contradicao, recarrega e tenta a 2a.
  let r = await jogarAdivinha(pagina, 'secreto');
  if (!r.ok && r.contradicao && recarregar) {
    await recarregar();
    r = await jogarAdivinha(pagina, 'palpite');
  }
  if (r.ok) evidencias.push(r.evidencia);
  else problemas.push(r.motivo);
}

// -- forca: so a letra clicada pode ser revelada -------------------------------

/** Estado da palavra: sequencia de letras/underscores mostrada na pagina. */
async function palavraDaForca(pagina) {
  const texto = await corpo(pagina);
  // Procura a linha com underscores (com ou sem espacos): "_ a _ _ a"
  const linhas = texto.split('\n').map((l) => l.trim());
  const linha = linhas.find((l) => /_/.test(l) && /^[\w_ ]+$/.test(l) && l.length >= 3);
  return linha ? linha.replace(/\s+/g, '') : null;
}

async function forca(pagina, problemas, evidencias) {
  const inicial = await palavraDaForca(pagina);
  if (!inicial || !inicial.includes('_')) {
    problemas.push('forca: nao achei a palavra escondida (linha com underscores) na pagina');
    return;
  }
  const tamanho = inicial.length;

  // Clica as 12 letras mais frequentes do portugues, uma por vez. Uma palavra
  // real em PT contem quase certamente varias delas — se NENHUMA revelar nada,
  // nao existe palavra secreta de verdade (e o que separa forca de fraude).
  let errosAntes = 0;
  let estadoAntes = inicial;
  let totalReveladas = 0;
  let clicou = false;
  for (const letra of ['a', 'e', 'o', 's', 'r', 'i', 'n', 'd', 'm', 'u', 't', 'c']) {
    const botao = pagina.locator('button', { hasText: new RegExp(`^\\s*${letra}\\s*$`, 'i') }).first();
    if ((await botao.count()) === 0) continue;
    const desabilitado = await botao.isDisabled().catch(() => false);
    if (desabilitado) continue;

    await botao.click({ timeout: 2000 }).catch(() => {});
    await pagina.waitForTimeout(200);
    clicou = true;

    const estado = await palavraDaForca(pagina);
    const textoBaixo = (await corpo(pagina)).toLowerCase();
    const erros = numeroPerto(textoBaixo, /erros?\D{0,15}?(\d+)/i) ?? errosAntes;
    const terminou = /ganhou|perdeu|venceu|fim de jogo|acabou|parab/.test(textoBaixo);

    if (!estado || !estado.includes('_')) {
      // palavra sumiu ou revelou inteira — tem que ser fim de jogo
      if (terminou) {
        if (totalReveladas === 0 && !/ganhou|venceu|parab/.test(textoBaixo)) {
          problemas.push(
            'forca: o jogo terminou em derrota sem NENHUMA das letras mais comuns do portugues revelar nada — a palavra secreta nao existe de verdade');
        } else {
          evidencias.push(`forca: jogo terminou apos clicar "${letra}"`);
        }
        return;
      }
      problemas.push(`forca: cliquei "${letra}" e a palavra com underscores sumiu sem o jogo terminar`);
      return;
    }
    if (estado.length !== tamanho) {
      problemas.push(
        `forca: a palavra tinha ${tamanho} posicoes e passou a ter ${estado.length} depois de clicar "${letra}" — o tamanho nao pode mudar`);
      return;
    }

    const reveladas = [];
    for (let i = 0; i < tamanho; i++) {
      if (estadoAntes[i] === '_' && estado[i] !== '_') reveladas.push(estado[i].toLowerCase());
    }

    if (reveladas.length > 0) {
      // Acerto: TUDO que apareceu tem que ser a letra clicada, e erros nao sobem.
      const intrusas = reveladas.filter((c) => c !== letra.toLowerCase());
      if (intrusas.length > 0) {
        problemas.push(
          `forca: cliquei "${letra}" e apareceram outras letras junto (${intrusas.join(',')}) — so posicoes de "${letra}" podiam ser reveladas`);
        return;
      }
      if (erros !== errosAntes) {
        problemas.push(`forca: cliquei "${letra}", a letra FOI revelada, mas o contador de erros subiu de ${errosAntes} pra ${erros}`);
        return;
      }
      totalReveladas += reveladas.length;
      evidencias.push(`forca: "${letra}" revelou ${reveladas.length} posicao(oes) corretamente`);
    } else {
      // Erro: contador tem que subir exatamente 1.
      if (erros !== errosAntes + 1) {
        problemas.push(
          `forca: cliquei "${letra}" (nao esta na palavra), nada foi revelado e o contador de erros foi de ${errosAntes} pra ${erros} — deveria subir exatamente 1`);
        return;
      }
      evidencias.push(`forca: "${letra}" errada, contador subiu ${errosAntes}→${erros} corretamente`);
    }
    errosAntes = erros;
    estadoAntes = estado;
    if (terminou) { evidencias.push('forca: jogo terminou'); break; }
  }
  if (!clicou) {
    problemas.push('forca: nao achei nenhum botao de letra clicavel — sem como jogar');
  } else if (totalReveladas === 0) {
    problemas.push(
      'forca: cliquei as 12 letras mais comuns do portugues e NENHUMA revelou posicao alguma — nao existe palavra secreta de verdade');
  }
}

// -- memoria: 16 cartas, duas viradas por vez, diferentes voltam a esconder ----

/** Acha o container com >=16 filhos do mesmo tipo (o tabuleiro). */
async function cartasDaMemoria(pagina) {
  return pagina.evaluate(() => {
    const pais = document.querySelectorAll('body *');
    for (const pai of pais) {
      const filhos = [...pai.children];
      if (filhos.length < 16) continue;
      const tags = new Set(filhos.map((f) => f.tagName));
      if (tags.size === 1) {
        // marca as cartas com um atributo pra gente clicar de fora
        filhos.forEach((f, i) => f.setAttribute('data-juiz-carta', String(i)));
        return filhos.length;
      }
    }
    return 0;
  });
}

async function estadoCartas(pagina) {
  // "revelada" = mostra texto/simbolo visivel diferente do verso padrao.
  return pagina.evaluate(() => {
    const cartas = [...document.querySelectorAll('[data-juiz-carta]')];
    const textos = cartas.map((c) => (c.innerText || '').trim());
    // o verso e o texto mais comum (geralmente vazio ou "?")
    const freq = {};
    textos.forEach((t) => { freq[t] = (freq[t] || 0) + 1; });
    const verso = Object.entries(freq).sort((a, b) => b[1] - a[1])[0][0];
    return textos.map((t) => (t === verso ? null : t));
  });
}

async function memoria(pagina, problemas, evidencias) {
  const n = await cartasDaMemoria(pagina);
  if (n < 16) {
    problemas.push(`memoria: esperava um tabuleiro com 16 cartas (mesmo tipo de elemento) e achei ${n}`);
    return;
  }

  const antes = await estadoCartas(pagina);
  const escondidasAntes = antes.filter((x) => x === null).length;
  if (escondidasAntes < 14) {
    problemas.push(`memoria: ${16 - escondidasAntes} cartas ja comecam reveladas — todas deviam comecar escondidas`);
    return;
  }

  // clica na primeira e na ultima carta escondidas (chance alta de serem diferentes)
  const idx = antes.map((v, i) => (v === null ? i : -1)).filter((i) => i >= 0);
  const [a, b] = [idx[0], idx[idx.length - 1]];
  await pagina.locator(`[data-juiz-carta="${a}"]`).click({ timeout: 2000 });
  await pagina.waitForTimeout(150);
  await pagina.locator(`[data-juiz-carta="${b}"]`).click({ timeout: 2000 });
  await pagina.waitForTimeout(200);

  const durante = await estadoCartas(pagina);
  const reveladas = durante.filter((x) => x !== null).length;
  if (reveladas !== 2) {
    problemas.push(
      `memoria: cliquei em 2 cartas e ${reveladas} ficaram reveladas — deviam ser exatamente 2`);
    return;
  }
  evidencias.push(`memoria: 2 cliques revelaram exatamente 2 cartas ("${durante[a]}" e "${durante[b]}")`);

  if (durante[a] !== durante[b]) {
    // par diferente tem que voltar a esconder (a maioria usa ~1s de timeout)
    await pagina.waitForTimeout(1600);
    const depois = await estadoCartas(pagina);
    const aindaReveladas = depois.filter((x) => x !== null).length;
    if (aindaReveladas !== 0) {
      problemas.push(
        `memoria: virei "${durante[a]}" e "${durante[b]}" (diferentes) e ${aindaReveladas} carta(s) continuam reveladas apos 1.6s — par errado deve voltar a esconder`);
      return;
    }
    evidencias.push('memoria: par diferente voltou a esconder sozinho');
  } else {
    evidencias.push('memoria: par igual permaneceu virado');
  }
}

// -- reflexo: tempo reportado plausivel, e repetivel ---------------------------

async function reflexo(pagina, problemas, evidencias) {
  // O jogo espera um tempo aleatorio e muda a tela; clicamos quando mudar.
  const inicioBody = await pagina.evaluate(() => document.body.style.backgroundColor || getComputedStyle(document.body).backgroundColor);

  // Se ha um botao de comecar, clica.
  const comecar = pagina.locator('button', { hasText: /come|inici|start|jogar/i }).first();
  if ((await comecar.count()) > 0) await comecar.click({ timeout: 2000 }).catch(() => {});

  // espera ate 6s por uma mudanca visual (cor de fundo de body ou de algum elemento grande)
  let mudou = false;
  for (let i = 0; i < 40; i++) {
    await pagina.waitForTimeout(150);
    const agora = await pagina.evaluate(() => document.body.style.backgroundColor || getComputedStyle(document.body).backgroundColor);
    if (agora !== inicioBody) { mudou = true; break; }
  }
  if (!mudou) {
    // fallback: alguns fazem a mudanca num alvo/div, nao no body — clica nele se existir
    const alvo = pagina.locator('#alvo, .alvo, #target, .target').first();
    if ((await alvo.count()) === 0) {
      problemas.push('reflexo: esperei 6s e a tela nunca mudou de cor — o sinal de "clique agora" nao aparece');
      return;
    }
  }

  await pagina.locator('body').click({ timeout: 2000 }).catch(() => {});
  await pagina.waitForTimeout(300);
  const texto = await corpo(pagina);
  const ms = numeroPerto(texto, /(\d+)\s*(?:ms|miliss|milis)/i);
  if (ms === null) {
    problemas.push(`reflexo: cliquei apos o sinal e a pagina nao mostrou o tempo em ms. Texto: ${JSON.stringify(texto.slice(0, 120))}`);
    return;
  }
  if (ms <= 0 || ms >= 10000) {
    problemas.push(`reflexo: tempo reportado ${ms}ms nao e plausivel (tem que ser >0 e <10000)`);
    return;
  }
  evidencias.push(`reflexo: tempo reportado ${ms}ms, plausivel`);
}

module.exports = {
  'adivinha.html': adivinha,
  'clique.html': reflexo,
  'forca.html': forca,
  'memoria.html': memoria,
};
