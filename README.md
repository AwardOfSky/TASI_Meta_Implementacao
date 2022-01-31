# TASI Meta de Implementaão
 
## Instruções:

Para correr o modelo proposto basta executar em linha de comando o nome do ficheiro Python principal "tf_gp_dcgan.py" seguido de um argumento para o dígito
a ser gerado pelo modelo e o nome da pasta para guardar os resultados da experiência.
Segue um comando exemplo:

```python
python tf_gp_dcgan.py 7 digito_7
```

Este comando cria uma pasta chamada *digito_7 com os resultados de evolução do modelo principal para o dígito 4, onde "*" é o prefixo da data da experiência. 
Ambos os argumentos são opcionais: o default para o primeiro argumento é o dígito 1 e se o nome da pasta não foi dada, o nome da pasta é simplesmente a data da experiência.

Para comparação, é também incluído um modelo Rede Adversarial Generativa Convolucional Profunda (DCGAN) convencional adaptado do site: https://www.tensorflow.org/tutorials/generative/dcgan.
Os parâmetros de input deste programa são os mesmo que os descritos para o modelo principal.
Comando exemplo:


```python
python dcgan.py 7 digito_7
```

Este comando cria uma pasta chamada *digito_7 com os resultados de evolução do modelo tradicional para o dígito 7.
Mais uma vez, "*" é o prefixo utilizado pela implementação.

É ainda disponibilizado um script linux (.sh) que corre todos os testes para todos os dois modelos e para todos os dígitos.
Este script tem o nome: "all_tests.sh".

Os requisitos das dependências Python para correr todos os modelos estão listados no ficheiro "requirements.txt".
Para instalar todos estes requisitos em Python basta introduzir o comando:

```bash
pip install -r requirements.txt
```

**Nota:** Será necessário uma versão de Python >= 3.7.


## Descrição do modelo:


O modelo implementado é uma variação de uma GAN tradicional onde, em vez de uma rede convolucional, o gerador consiste numa população de expressões simbólicas evoluídas com recurso a Programação Genética (GP).
Mais concretamente trata-se de uma adaptação de uma DCGAN, onde a componente do discriminador continua a ser uma Rede Neuronal Convolucional (CNN) normal e o gerador um processo evolucionário onde usamos o TensorGP.
O TensorGP é um motor de GP desenvolvido no âmbito do meu mestrado, disponível publicamente no seguinte repositório: https://github.com/AwardOfSky/TensorGP.
A principal vantagem de TensorGP é conseguir acelarar a fase de avaliação do domínio em GP recorrendo a técnicas de paralelização e de reutilização de resultados de aptidão intermediários [1, 2].

Na configuração atual do modelo, cada passo de treino pode ser descrito pelo seguinte algoritmo.
Primeiro retiramos um batch de imagens reais do dataset. De seguida evoluímos um segundo batch de expressões simbólicas com recurso ao TensorGP por n gerações.
A run evolucionário terá como função de aptidão um forward pass pelo discriminador atual. É importante notar que ambos que como o discriminador vai também evoluindo ao longo do tempo,
a fitness dos indivíduos muda a cada passo de treino.
Depois de gerados os dois batches de imagens, estes são passados ao discriminador para treino normal por retro-propagação.
Finalmente, após o treino do gerador, as losses do modelo são calculadas.
Cada passo de treino é repetido n vezes para cada época (epoch).
O algortimo pode ser resumido no seguinte psedocódigo:


```python
for each trainin_step:
	
	get real batch from dataset
	
	# fitness is a forward pass of the discriminator
	generate batch from GP run with n generations
	
	train discriminator with real batch
	train discriminator with generated batch
	
	calculate losses
```


Concretamente, para a implementação foi usado Python juntamente com a biblioteca TensorFlow e o backend Keras para ajudar na definição das camadas convolucionais da rede do discriminador.
O TensorFlow é também utilizado para lidar com as operações tensoriais definidas pelo TensorGP.


## Experimentação

A experimentação para este trabalho incidiu no treino do nosso modelo com os dígitos do dataset MNIST.
A ideia inicial era a de testar para todos os dígitos simultaneamente. Nesse sentido preparámos um setup experimental onde é evoluído todo o conjunto de dados por 5 épocas,
com um tamanho de batch de 32 e com 50 gerações para cada passo de treino do gerador.
Foi ainda implementado uma espécie de meta-elitismo no gerador que possibilita a inclusão dos n melhores indíviduos no próximo passo de treino do gerador.
Para os testes realizados neste trabalho foi considerado n = 1.
Desta maneira, a população da primeira geração (no primeiro passo de treino) é gerada aleatóriamente, de acordo com os parâmetros de profundidade dos indivíduos e modo de geração, e passos subsequentes têm a sua
população inicial gerada também aleatoriamente com exceção de 1 indíviduo que corresponde à melhor aptidão do passo de treino anterior.


Os resultados do treino mostram claramente que é possível evoluir dígitos mais simples (como o 0 e o 8).

[mostrar imagens]

No entanto, algumas das populações geradas nesta fase inicial mostraram-se também bastante aleatórias.

[mostrar mais imagens As e space invaders]

A primeira hipótese considerada para este resultado prende-se com o facto de terem sido evoluídos todos os dígitos.
Para testar esta hipótese, consideramos o mesmo setup experimental, mas em vez de serem testados todos os dígitos de uma só vez, cada dígito foi testado individualmente.
Neste último teste os resultados foram consideravelmente melhores já que foram evoluídos artifactos para quase todos os dígitos com relativo sucesso.

[mostrar mais imagens boas]

Numa fase posterior, e para completar a fase experimental, o modelo proposto foi comparado a uma DCGAN normal:

[resultados de avaliação para DCGAN]

Para o modelo DCGAN pode-se verificar que 5 épocas não é suficiente para gerar todos os dígitos.
Comparação da loss do discriminador para os dois modelos:

[imagem loss]


## Conclusão:

A evolução de expressões simbólicas em modelos adversariais é algo bastante inexplorado na literatura até à data.
Este trabalho serve como uma prova de conceito do facto de ser possível evoluir dígitos 

Os resultados mostram claramente que, ao passo que em 5 épocas não é suficiente para serem gerados dígitos no modelo original,
no modelo proposto é possível alcançar resultados bastante encorajadores.

No entanto, como podemos comprovar ao correr o modelo, o tempo de execução para o modelo proposto é bastante maior já que a evolução de expressões é uma tarefa computacionalmente dispendiosa.
No entanto, face a estes resultados preliminares, podemos concluir que este modelo é uma abordagem que deve ser considerada futuramente.
Para além do mais, com avanços em técnicas de paralelização de hardware, é possível que Computação Evolucionária volte a ser a técnica preferencial para a geração de imagens.

No que toca a trabalho futuro, uma funcionalidade crucial no modelo, e o próximo passo no desenvolvimento deste projeto prende-se com a incorporação de uma solução para agragar as expressões simbólicas a serem evoluídas num espaço latente organizado, possivelmente
com recurso a arquivos de soluções ou a autoencoders variacionais (ver artigo da primeira meta).


## Referências:

[1] - Francisco Baeta, João Correia, Tiago Martins, and Penousal Machado. Tensorgp —
genetic programming engine in tensorflow. In Applications of Evolutionary Computation – 24th International Conference, EvoApplications 2021, pages 763–778.
Springer,2021.

[2] - Francisco Baeta, João Correia, Tiago Martins, and Penousal Machado. Speed benchmarking of genetic programming frameworks. In GECCO 2021 - Proceedings ofthe
2021 Genetic and Evolutionary Computation Conference, page to appear. ACM,2021.
