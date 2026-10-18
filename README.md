# Local Coder

Um agente de código que roda inteiro no seu PC, sem nuvem e sem custo. Você
descreve a tarefa e ele lê a pasta do projeto, escreve o código, compila,
testa, instala dependências e corrige os erros até funcionar.

- **Motor:** llama.cpp (`llama\llama-server.exe`, build b11711, CUDA 13).
- **Modelos:** Qwen3.6-35B-A3B em duas versões, em `models\`:
  - normal (`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, unsloth): a melhor qualidade.
  - abliterated (`Huihui-Qwen3.6-35B-A3B-abliterated-Q4_K.gguf`, huihui-ai):
    sem recusas, um pouco menos precisa. Ela traz cabeças MTP, então roda com
    decodificação especulativa (`--spec-type draft-mtp`).
- **Interface:** VertexUI (Dear ImGui).

Dois perfis, trocáveis no topo da barra lateral (mudam o foco do agente e a cor):

- **Red Hand** — ferramentas para jogos: mods, engenharia reversa, hooks/detours, scripts de jogo. O agente sabe a fronteira do que dá para testar aqui e do que só roda no jogo.
- **Geral** — software do dia a dia, sem relação com jogos.

## Instalação

O código vive no repositório; os pesos do modelo (~44 GB) e a build do llama.cpp **não** vão aqui — você baixa uma vez.

1. **Dependências Python** (precisa de Python 3.11+ e uma GPU NVIDIA):

   ```bat
   pip install -r requirements.txt
   ```

2. **llama.cpp** (build CUDA para Windows) — baixe de
   [github.com/ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases)
   o `llama-*-bin-win-cuda-*-x64.zip` mais o `cudart-*-win-cuda-*-x64.zip`
   correspondente, e extraia os dois para a pasta `llama\` ao lado do `run.py`
   (tem que existir `llama\llama-server.exe`).

3. **Modelos** (GGUF) para a pasta `models\`:
   - normal: [`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) (unsloth) + `mmproj-F16.gguf` renomeado para `mmproj-qwen3.6-F16.gguf`;
   - abliterated: [`...abliterated-ggml-model-Q4_K.gguf`](https://huggingface.co/huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF) renomeado para `Huihui-Qwen3.6-35B-A3B-abliterated-Q4_K.gguf` + `mmproj-model-f16.gguf` renomeado para `mmproj-huihui-abliterated-f16.gguf`.

   Os nomes esperados e os caminhos estão em `lcoder/server.py` (lista `MODELS`); ajuste lá se usar outro quant.

Pensado para uma RTX 5070 (12 GB) + 64 GB de RAM: o modelo é MoE (3B ativos) e o llama.cpp mantém parte dos experts na RAM (`-ncmoe`). Em outra placa, ajuste `NCMOE` em `lcoder/server.py` ou deixe o `--fit` automático.

## Rodar

Dê dois cliques em **`Local Coder.bat`**, ou:

```bat
python run.py
```

O modelo começa a carregar sozinho ao abrir a janela (20 s a 1 min). Escolha
a pasta do projeto na coluna da esquerda, escreva a tarefa e aperte Enter.

Para uma tarefa sem janela, totalmente autônoma:

```bat
python run.py --cli --workspace C:\meu\projeto "crie X, compile e teste"
```

## Autonomia

| modo | o que faz sem perguntar |
|---|---|
| Autônomo total | tudo: editar, rodar comandos, instalar pacotes |
| Pergunta antes de comandos | edita arquivos; pede ok antes de cada comando |
| Pergunta tudo | pede ok para cada edição e cada comando |

No modo total ele roda qualquer comando do PowerShell com as suas permissões.
Dê a ele uma pasta de projeto própria, e não a raiz de um disco.

## Ferramentas do agente

`list_dir`, `read_file`, `search`, `write_file`, `edit_file`, `run`
(PowerShell, com timeout), `bg_start`/`bg_output`/`bg_stop` (servidores e
apps que não terminam) e `fetch_url` (documentação na web).

Ao abrir, ele detecta os compiladores instalados (Lua, luacheck, gcc/g++,
nasm, cmake, Python, Node, Rust, .NET, MSVC) e coloca a lista no prompt de
sistema, para nunca adivinhar o que existe.

## Arquivos

```
lcoder/server.py   inicia e para o llama-server, lê o log
lcoder/llm.py      cliente de chat em streaming (só stdlib)
lcoder/tools.py    as ferramentas
lcoder/prompt.py   prompt de sistema + detecção de compiladores
lcoder/agent.py    o loop do agente, aprovações, compactação de contexto
lcoder/app.py      a janela
lcoder/cli.py      modo sem janela
sessions/          cada conversa salva em JSON
```

## Ajuste de memória

`--fit` (ligado por padrão no llama.cpp) decide quantas camadas de experts
ficam na RAM para o resto caber nos 12 GB da RTX 5070. O cache de contexto
usa q8_0, o que permite 64k tokens. Contextos maiores (96k/128k) funcionam,
mas jogam mais experts para a CPU e ficam mais lentos.
