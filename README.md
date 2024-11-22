
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
$ python Results_accuracy/graph.py I1/20241120-142131-WISARD-accuracy.csv I1/20241120-142909-BTHOWeN-accuracy.csv I1/20241120-141214-ULEEN-accuracy.csv I1-accuracy.png
$ python Results_accuracy/graph.py I2/20241120-142255-WISARD-accuracy.csv I2/20241120-142928-BTHOWeN-accuracy.csv I2/20241120-141338-ULEEN-accuracy.csv I2-accuracy.png
$ python Results_accuracy/graph.py M1/20241120-142401-WISARD-accuracy.csv M1/20241120-142948-BTHOWeN-accuracy.csv M1/20241120-141455-ULEEN-accuracy.csv M1-accuracy.png
$ python Results_accuracy/graph.py M2/20241120-142527-WISARD-accuracy.csv M2/20241120-143014-BTHOWeN-accuracy.csv M2/20241120-141638-ULEEN-accuracy.csv M2-accuracy.png
$ python Results_accuracy/graph.py S1/20241120-142654-WISARD-accuracy.csv S1/20241120-143127-BTHOWeN-accuracy.csv S1/20241120-141821-ULEEN-accuracy.csv S1-accuracy.png
$ python Results_accuracy/graph.py S2/20241120-142821-WISARD-accuracy.csv S2/20241120-143148-BTHOWeN-accuracy.csv S2/20241120-141957-ULEEN-accuracy.csv S2-accuracy.png
```