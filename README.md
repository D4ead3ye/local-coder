# Local Coder

Um **hub de IA que roda inteiro no seu PC**, sem nuvem e sem custo. Começou como
um agente de código e virou uma casa para várias ferramentas locais, todas na
mesma janela:

- **Agente** — você descreve a tarefa e ele lê a pasta do projeto, escreve o
  código, compila, testa, instala dependências e corrige os erros até funcionar.
- **Áudio** — gera SFX e música a partir de um texto (Stable Audio Open), mais
  um motor procedural instantâneo para cues de UI.
- **Imagem** — geração de imagem local na GPU (SDXL e variantes).

- **Motor do agente:** llama.cpp (`llama\llama-server.exe`, build b11711, CUDA 13).
- **Interface:** VertexUI (Dear ImGui) — a aparência inteira troca por tema
  (ENA World retrô, YMO, Y2K, Frutiger Aero e outros), no card **visual** da
  barra lateral do Agente.

Pensado para uma **RTX 5070 (12 GB) + 64 GB de RAM**. Em outra placa, dá para
ajustar (veja *Ajuste de memória*).

---

## 1. Instalação base (o Agente)

O código vive no repositório; os pesos do modelo (~44 GB) e a build do llama.cpp
**não** vão aqui — você baixa uma vez.

**Pré-requisitos:** Python 3.11+ e uma GPU NVIDIA.

1. **Clone e dependências Python:**

   ```bat
   git clone https://github.com/D4ead3ye/local-coder
   cd local-coder
   pip install -r requirements.txt
   ```

   > A UI (VertexUI) é puxada do GitHub pelo `requirements.txt`. Se você mantém
   > um checkout local do `vertexui-repo` ao lado, o `run.py` o coloca na frente.

2. **llama.cpp** (build CUDA para Windows) — baixe de
   [github.com/ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases)
   o `llama-*-bin-win-cuda-*-x64.zip` mais o `cudart-*-win-cuda-*-x64.zip`
   correspondente, e extraia os dois para a pasta `llama\` ao lado do `run.py`
   (tem que existir `llama\llama-server.exe`).

3. **Modelos** (GGUF) para a pasta `models\`:

   | chave | arquivo | onde pegar |
   |---|---|---|
   | normal | `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` + `mmproj-qwen3.6-F16.gguf` | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) |
   | abliterated | `Huihui-Qwen3.6-35B-A3B-abliterated-Q4_K.gguf` + `mmproj-huihui-abliterated-f16.gguf` | [huihui-ai/...abliterated-MTP-GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF) |

   Os nomes e caminhos exatos estão em `lcoder/server.py` (lista `MODELS`);
   renomeie os arquivos para bater, ou ajuste lá se usar outro quant. O `mmproj`
   é o que dá visão ao modelo (replicar UI de um print).

Só com isso o **Agente** já roda. Áudio e imagem são um passo opcional à parte.

---

## 2. Áudio e imagem (venv neural, opcional)

As ferramentas de áudio e imagem usam torch/diffusers na GPU. Para manter o app
leve, elas vivem num **venv separado** (`ai-venv\`) que roda como um servidor,
igual ao llama-server. Um comando só provisiona tudo:

```bat
python -m lcoder.neural_setup
```

Isso cria o `ai-venv\`, instala o torch da build CUDA e o resto do
`requirements-neural.txt`. Rode `python -m lcoder.neural_setup --check` para ver
o estado.

Os pesos baixam na **primeira geração** (ou seja, não ocupam espaço até usar):

- **Imagem** — SDXL 1.0 (~7 GB, não exige login) baixa sozinho. Também há SDXL
  Turbo (rápido, poucos passos) e SD 1.5. Escolha no card **modelo** da aba.
- **Áudio** — Stable Audio Open 1.0 faz música (até ~47 s) e SFX. É *gated* no
  Hugging Face: aceite a licença na página do modelo e faça
  `huggingface-cli login` uma vez; depois baixa sozinho. O motor **procedural**
  não precisa de nada disso (é instantâneo, sem GPU).

As imagens geradas ficam em `output\images\`.

---

## 3. Rodar

Dê dois cliques em **`Local Coder.bat`**, ou:

```bat
python run.py
```

Abre a **home do hub**, com um bloco para cada ferramenta. No **Agente**, o
modelo começa a carregar sozinho (20 s a 1 min); escolha a pasta do projeto na
esquerda, escreva a tarefa e aperte Enter.

Para uma tarefa sem janela, totalmente autônoma:

```bat
python run.py --cli --workspace C:\meu\projeto "crie X, compile e teste"
```

### Perfis do agente

Dois perfis, trocáveis no topo da barra lateral (mudam o foco e a cor):

- **Red Hand** — ferramentas para jogos: mods, engenharia reversa, hooks/detours,
  scripts de jogo. O agente sabe a fronteira do que dá para testar aqui e do que
  só roda no jogo.
- **Geral** — software do dia a dia.

### Autonomia

| modo | o que faz sem perguntar |
|---|---|
| Autônomo total | tudo: editar, rodar comandos, instalar pacotes |
| Pergunta antes de comandos | edita arquivos; pede ok antes de cada comando |
| Pergunta tudo | pede ok para cada edição e cada comando |

No modo total ele roda qualquer comando do PowerShell com as suas permissões.
Dê a ele uma pasta de projeto própria, e não a raiz de um disco.

---

## Gerar o .exe

Para um app que roda sem Python instalado:

```powershell
.\build.ps1
```

Gera `dist\LocalCoder\LocalCoder.exe`. Copie as pastas `llama\` e `models\` (e
`ai-venv\`, se quiser áudio/imagem no .exe) para dentro de `dist\LocalCoder\`
antes de rodar — os pesos não são embutidos.

---

## Ferramentas do agente

`list_dir`, `read_file`, `search`, `write_file`, `edit_file`, `run`
(PowerShell, com timeout), `bg_start`/`bg_output`/`bg_stop` (servidores e apps
que não terminam), `screenshot`/`view_image` (visão) e `fetch_url`
(documentação na web).

Ao abrir, ele detecta os compiladores instalados (Lua, luacheck, gcc/g++, nasm,
cmake, Python, Node, Rust, .NET, MSVC) e coloca a lista no prompt de sistema,
para nunca adivinhar o que existe.

## Arquivos

```
lcoder/server.py        inicia e para o llama-server, lê o log
lcoder/llm.py           cliente de chat em streaming (só stdlib)
lcoder/tools.py         as ferramentas do agente
lcoder/prompt.py        prompt de sistema + detecção de compiladores
lcoder/agent.py         o loop do agente, aprovações, compactação de contexto
lcoder/app.py           a janela (hub, todas as abas)
lcoder/cli.py           modo sem janela
lcoder/audio.py         controlador de áudio (procedural + neural)
lcoder/image.py         controlador de imagem (SDXL via venv neural)
lcoder/neural_setup.py  provisiona o ai-venv (torch/diffusers)
lcoder/sfx/             motor de áudio (DSP numpy) + servidores neurais
sessions/               cada conversa salva em JSON
output/images/          imagens geradas
```

## Ajuste de memória

`--fit` (ligado por padrão no llama.cpp) decide quantas camadas de experts ficam
na RAM para o resto caber nos 12 GB da RTX 5070. O cache de contexto usa q8_0, o
que permite 64k tokens. Contextos maiores (96k/128k) funcionam, mas jogam mais
experts para a CPU e ficam mais lentos. Em outra placa, ajuste `NCMOE` em
`lcoder/server.py` ou deixe o `--fit` automático.
