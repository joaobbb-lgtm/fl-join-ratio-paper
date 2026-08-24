# Execução sincronizada

O script `run_synchronized_experiment.sh` inicia a Tapo/telemetria no ambiente
`telemetria`, aguarda a primeira amostra e só então inicia o PFLlib no ambiente
`fl-dissertacao`. Ao final do treinamento, encerra a coleta de forma controlada e
gera o resumo da telemetria.

Exemplo:

```bash
ALGORITHM=FedAvg JOIN_RATIO=0.25 REPETITION=1 \
  experiment_scripts/run_synchronized_experiment.sh
```

Por padrão, as credenciais são carregadas automaticamente de
`~/.config/fl-telemetry/kasa.env`, criado durante a preparação da telemetria.
Outro arquivo pode ser selecionado com `KASA_ENV_FILE`. As credenciais nunca
são copiadas para a pasta do experimento.

As opções podem ser fornecidas como variáveis de ambiente: `ALGORITHM`,
`DATASET`, `MODEL`, `DEVICE`, `GLOBAL_ROUNDS`, `NUM_CLIENTS`, `JOIN_RATIO`,
`LOCAL_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `REPETITION`,
`TELEMETRY_INTERVAL`, `EXPERIMENT_ID` e `OUTPUT_ROOT`.

Cada execução cria `experiments/<EXPERIMENT_ID>/` com:

- `metadata.env`: configuração, horários sincronizados e código de saída;
- `pfllib.log`: saída completa do treinamento;
- `telemetry.csv`: amostras da Tapo, CPU, RAM e RTX;
- `telemetry.log`: saída operacional do coletor;
- `telemetry_summary.txt`: resumo e energia integrada.

As credenciais continuam sendo lidas exclusivamente de `KASA_USERNAME` e
`KASA_PASSWORD`; elas não são copiadas para os metadados.
