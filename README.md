# Driving World Model

Proyecto final de Deep Reinforcement Learning -- Comparativa empírica entre un baseline model-free (DQN/Rainbow) y un World Model (DreamerV3 / R2-Dreamer) en tareas de control continuo visual con MuJoCo.

## Contexto del proyecto

En la primera parte del curso implementamos DQN y Rainbow DQN, que aprenden directamente una Q-function sobre observaciones RGB discretizando el espacio de acciones. El objetivo de esta segunda fase es incorporar un **World Model**: un modelo que aprende una representación latente del entorno y simula trayectorias futuras *en imaginación*, permitiendo entrenar una política sin necesidad de interactuar constantemente con el entorno real. Esto mejora la eficiencia muestral, especialmente cuando la interacción con el entorno es costosa.

## Repo utilizado: R2-Dreamer

Usamos el repositorio [NM512/r2dreamer](https://github.com/NM512/r2dreamer), que incluye:

- Una **reproducción eficiente de DreamerV3** en PyTorch (~5x más rápida que el antiguo [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)).
- **R2-Dreamer** (ICLR 2026): una variante que elimina el decoder de reconstrucción de imágenes y lo sustituye por una pérdida de redundancia reducida (Barlow Twins-style), consiguiendo un ~1.6x speedup adicional.
- Otros baselines: InfoNCE, DreamerPro.

Se selecciona el algoritmo con un solo flag: `model.rep_loss=r2dreamer|dreamer|infonce|dreamerpro`.

## Estructura del repositorio

```
  driving-world-model/
  ├── pyproject.toml          # uv dependencies
  ├── train.py                # Punto de entrada Hydra
  ├── configs/
  │   └── config.yaml         # Config unificada (env + model + training)
  ├── src/
  │   ├── __init__.py
  │   ├── agent.py            # Dreamer: world model + R2-Dreamer Barlow loss + actor-critic        
  │   ├── rssm.py             # RSSM: Block-GRU + observe/imagine/kl_loss
  │   ├── networks.py         # ConvEncoder, MLPHead, Projector, BlockLinear, ReturnEMA
  │   ├── distributions.py    # OneHotDist, TwoHot, BoundedNormal, symlog, kl
  │   ├── buffer.py           # Replay buffer (TorchRL SliceSampler)
  │   ├── envs.py             # DMC wrapper + ParallelEnv + make_envs
  │   └── tools.py            # Logger, weight_init, utilities
  └── r2dreamer/              # (repo original, intacto)
```

  Para ejecutar:

  cd driving-world-model

  # Usar Python 3.11 (dm-control/labmaze no es compatible con Python 3.14 en Windows)
  uv python install 3.11
  uv venv --python 3.11

  # Instalar deps con uv
  uv sync

  # Debug rápido (pocos steps)
  uv run python train.py env.task=dmc_walker_walk training.steps=10000 env.env_num=2 env.eval_episode_num=2

  # Entrenamiento completo
  uv run python train.py env.task=dmc_walker_walk

## Estructura del repositorio `r2dreamer/`

```
r2dreamer/
├── train.py              # Punto de entrada (usa Hydra para configuración)
├── dreamer.py            # Clase Dreamer: world model + actor-critic
├── rssm.py               # RSSM: modelo de dinámicas latente (corazón del world model)
├── networks.py           # Redes: encoder CNN, decoder, MLP heads, BlockLinear, Projector
├── distributions.py      # Distribuciones: OneHotDist, TwoHotSymlog, BoundedNormal, etc.
├── buffer.py             # Replay buffer basado en TorchRL (SliceSampler por episodios)
├── trainer.py            # OnlineTrainer: loop de entrenamiento y evaluación
├── tools.py              # Utilidades: logger, symlog, seeds, optimizador, etc.
├── optim/                # Optimizadores custom
│   ├── agc.py            # Adaptive Gradient Clipping
│   └── laprop.py         # Optimizador LaProp (reemplaza Adam)
├── configs/
│   ├── configs.yaml      # Config raíz Hydra (defaults, buffer, trainer)
│   ├── env/
│   │   ├── dmc_vision.yaml   # DMC con imágenes 64x64 (nuestro caso)
│   │   ├── dmc_proprio.yaml  # DMC con estados vectoriales
│   │   ├── atari100k.yaml    # Atari
│   │   ├── crafter.yaml      # Crafter
│   │   ├── metaworld.yaml    # MetaWorld (manipulación robótica)
│   │   └── memorymaze.yaml   # Memory Maze
│   └── model/
│       ├── _base_.yaml       # Hiperparámetros base (RSSM, encoder, decoder, actor, critic, etc.)
│       ├── size12M.yaml      # Modelo 12M params (suficiente para DMC, funciona en GPU de consumo)
│       ├── size25M.yaml      # Modelo 25M params
│       ├── size50M.yaml      # ...
│       ├── size100M.yaml
│       ├── size200M.yaml
│       └── size400M.yaml
├── envs/
│   ├── __init__.py       # Factory: make_envs() y make_env() según la suite
│   ├── dmc.py            # Wrapper DeepMind Control Suite (Gymnasium, imágenes 64x64)
│   ├── dmc_subtle.py     # DMC con objetos pequeños (benchmark extra)
│   ├── atari.py          # Wrapper Atari
│   ├── crafter.py        # Wrapper Crafter
│   ├── metaworld.py      # Wrapper MetaWorld
│   ├── memorymaze.py     # Wrapper Memory Maze
│   ├── parallel.py       # Ejecución paralela de entornos (multiprocessing)
│   └── wrappers.py       # Wrappers genéricos: TimeLimit, NormalizeActions, OneHotAction, Dtype
├── runs/                 # Scripts de lanzamiento por benchmark
│   ├── dmc.sh            # Lanza todos los tasks de DMC con múltiples seeds
│   ├── atari.sh
│   ├── crafter.sh
│   ├── metaworld.sh
│   └── memorymaze.sh
└── docs/
    ├── docker.md         # Instrucciones Docker
    └── tensor_shapes.md  # Guía de shapes de tensores (muy útil para entender el código)
```

python train.py env=dmc_vision env.task=dmc_walker_walk model.compile=False trainer.steps=1e4     
  env.env_num=2 env.eval_episode_num=2 logdir=./logdir/debug

## Cómo encajan las piezas

### 1. `train.py` -- Punto de entrada

Usa Hydra para cargar la configuración (env + model), crea el buffer, los entornos paralelos, instancia el agente `Dreamer` y lanza el `OnlineTrainer`. Al terminar guarda los pesos en `latest.pt`.

### 2. `dreamer.py` -- El agente (World Model + Actor-Critic)

Contiene toda la lógica del agente en una sola clase `Dreamer(nn.Module)`. Componentes:

- **World Model**:
  - `encoder`: CNN que mapea imágenes RGB 64x64 a embeddings latentes.
  - `rssm`: el modelo de dinámicas (RSSM), que mantiene un estado latente = determinista (GRU con BlockLinear) + estocástico (categoricals con unimix). Puede hacer `observe()` (con observación real) o `imagine()` (sin observación, solo con acciones).
  - `reward`: MLP head que predice la recompensa desde el estado latente (distribución TwoHotSymlog con 255 bins).
  - `cont`: MLP head que predice si el episodio continúa (distribución binaria).
  - `decoder` (solo en modo `dreamer`): CNN transpuesta que reconstruye la imagen. En R2-Dreamer se elimina y se usa un `Projector` con pérdida Barlow Twins.

- **Actor-Critic** (entrenados en imaginación):
  - `actor`: genera acciones continuas (distribución BoundedNormal) desde el estado latente imaginado.
  - `value`: estima el valor con lambda-returns y distribución TwoHotSymlog.
  - `_slow_value`: target network del critic (EMA update).
  - `ReturnEMA`: normalización de retornos por percentiles (truco clave de DreamerV3).

El método `update()` hace un paso completo de entrenamiento: samplea del buffer, observa con el RSSM, calcula pérdidas del world model (KL + reward + cont + recon/barlow), imagina trayectorias de 15 pasos, y entrena actor-critic sobre ellas.

### 3. `rssm.py` -- El modelo de dinámicas latente

El corazón de Dreamer. El estado latente tiene dos componentes:
- **Determinista** (`deter`): mantenido por un Block-GRU (GRU con BlockLinear para eficiencia). Tamaño 2048 en el modelo 12M.
- **Estocástico** (`stoch`): 32 variables categóricas de 16 clases cada una, con unimix para evitar colapso.

Dos modos de operación:
- `observe(embed, action, is_first)`: dado el embedding real del encoder, computa el *posterior* (lo que realmente pasó) y el *prior* (lo que el modelo predecía). La diferencia KL entre ambos es la señal de entrenamiento.
- `imagine(action, initial_state)`: genera trayectorias latentes usando solo el prior, sin observaciones. Esto es lo que permite entrenar el actor-critic "en imaginación".

### 4. `networks.py` -- Redes neuronales

- `MultiEncoder`: CNN con `Conv2dSamePad` (kernels 5x5, depth creciente) para imágenes + MLP para estados vectoriales.
- `MultiDecoder`: CNN transpuesta para reconstruir imágenes (solo en modo `dreamer`).
- `MLPHead`: red genérica usada para reward, cont, actor y critic. Cada head tiene su distribución de salida configurable.
- `BlockLinear`: capa lineal por bloques (eficiencia de memoria y cómputo en el GRU del RSSM).
- `Projector`: usado en R2-Dreamer para la pérdida Barlow Twins en lugar del decoder.
- `ReturnEMA`: normalización de retornos por percentiles 5%-95%.

### 5. `buffer.py` -- Replay Buffer

Basado en `torchrl.ReplayBuffer` con `SliceSampler` que muestrea secuencias temporales contiguas dentro de episodios. Almacena las observaciones, acciones y los estados latentes del RSSM (para poder re-inicializar el RSSM sin re-observar toda la secuencia).

### 6. `trainer.py` -- Loop de entrenamiento

`OnlineTrainer.begin()` ejecuta el loop principal:
1. Stepea los entornos en CPU (para evitar sincronizaciones GPU<->CPU).
2. Mueve observaciones a GPU con `non_blocking=True`.
3. El agente actúa (`agent.act()`), que hace un paso de RSSM + política.
4. Guarda transiciones en el buffer.
5. Cuando hay suficientes datos, llama a `agent.update()` (entrena world model + actor-critic).
6. Periódicamente evalúa y loggea en TensorBoard.

## Diferencias clave entre DreamerV3 y R2-Dreamer

| | DreamerV3 (`dreamer`) | R2-Dreamer (`r2dreamer`) |
|---|---|---|
| Representación | Reconstruye imágenes con decoder CNN | Sin decoder; usa Projector + pérdida Barlow Twins |
| Velocidad | Baseline (ya ~5x más rápido que dreamerv3-torch) | ~1.6x más rápido que el baseline |
| Rendimiento | State-of-the-art en DMC | Comparable o superior, sin reconstruir imágenes |
| VRAM | Mayor (decoder CNN es costoso) | Menor |

## Cómo ejecutar (DMC Vision)

```bash
cd r2dreamer

# Instalar dependencias
pip install -r requirements.txt

# Entrenar DreamerV3 en walker_walk con imágenes
python train.py env=dmc_vision env.task=dmc_walker_walk model.rep_loss=dreamer logdir=./logdir/dreamer_walker

# Entrenar R2-Dreamer (sin decoder) en walker_walk
python train.py env=dmc_vision env.task=dmc_walker_walk model.rep_loss=r2dreamer logdir=./logdir/r2dreamer_walker

# Monitorizar
tensorboard --logdir ./logdir
```

El modelo por defecto es `size12M` (12M de parámetros), suficiente para DMC y funciona con ~8GB de VRAM.

## Dependencias principales

- PyTorch 2.8, TorchRL 0.9.2
- MuJoCo 3.3, dm_control 1.0.28
- Gymnasium 1.2.1
- Hydra 1.3.2 (configuración)
- NumPy 1.26, OpenCV

## Referencias

- [DreamerV3: Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) (Hafner et al., 2023)
- [R2-Dreamer: Redundancy-Reduced World Models](https://openreview.net/forum?id=Je2QqXrcQq) (Morihira et al., ICLR 2026)
- [World Models](https://arxiv.org/abs/1803.10122) (Ha & Schmidhuber, 2018)
