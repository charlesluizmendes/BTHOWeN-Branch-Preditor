
*Todo o trabalho aqui desenvolvido foi inspirado no https://github.com/LaqVillon/Dissertation-branch-predictor, como possível solução para execução de algoritmos como BTHOWeN, ULEEN e DWN para Preditores de Dewsvio.

# Preparação do Ambiente

Para instalar todas as dependências do projeto execute o comando abaixo:

```
$ pip install -r requirements.txt
```

# Execução dos Algoritmos

### Wisard

Exemplo:

```
$ python wisard.py Dataset_pc_decimal/M1.txt 32 2 2 2 2 2 3 4 1 1
```

### BTHOWeN

Exemplo:
 
```
$ python bthowen.py Dataset_pc_decimal/I1.txt 32 2 2 2 2 2 3 4 1 1
```

### ULEEN

Exemplo:

```
$ python uleen.py Dataset_pc_decimal/S2.txt 32 2 2 2 2 2 3 4 1 1
```

# Criar graficos dos resultados

Para criar os graficos do resultado das execuções anteriores, execute os comandos abaixo:

```
$ cd Results_accuracy
```

```
$ python .\graph.py .\I1
$ python .\graph.py .\I2
$ python .\graph.py .\M1
$ python .\graph.py .\M2
$ python .\graph.py .\S1
$ python .\graph.py .\S2
```