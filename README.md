# TG1 - 5º semestre de BD

 

Professor da Disciplina: Giuliano Bertoti 

 

# TG

 

Aluno: Rodrigo Prado da Silva - ra: 1460281813035

Orientador: Giuliano Bertoti.

 

Título do TG: UTILIZAÇÃO DE MACHINE LEARNING PARA REVISÃO DE DOCUMENTOS E PESQUISAS JURÍDICAS 


 

#### 1. INTRODUÇÃO
Com o avanço tecnológico cada vez mais profissões estão sendo facilitadas ou substituídas por tecnologia. Dado que muitas tarefas repetitivas e exaustivas tendem a levar muito tempo e necessitar de muito esforço para que pessoas possam realiza-las, além disso pessoas podem ter diferentes fatores que podem mudar o resultado da tarefa, como por exemplo cansado, sono, fome, humor, stress e vários outros fatores, sendo assim mais propenso à cometer erros, onde tecnologias estão menos propensas a cometer tais erros, além de poder trabalhar sem descanso e ainda ser mais rápido e eficiente na realização das tarefas.
Por essa e outras razões é previsto que um terço dos empregos existentes hoje poderia ser ocupado por tecnologias, inteligência artificial, robótica e algoritmos até 2025 (Brougham,2017). Principalmente em trabalhos e tarefas repetitivas onde há padrões nas realizações dessas tarefas, desse modo, uma profissão que pode ser muito aprimorada e ter sua produtividade e eficiência melhorada pela tecnologia, inteligência artificial, e automatização de algumas de suas tarefas é a área advocacia.
A previsão é que o trabalho rotineiro em fábricas e escritórios, como contabilidade ou operação de máquinas básicas, é mais vulnerável à automação (Miller, 2016). Mas o software de IA que pode ler e analisar texto ou fala - o chamado processamento de linguagem natural - está invadindo o trabalho dos profissionais. Por exemplo, há muito trabalho legal rotineiro. Mas esse trabalho de rotina, examinando documentos para obter informações relevantes, é envolto em linguagem que protegia os advogados dos efeitos da automação. Mas não mais.
Na advocacia o Machine Learning pode ser usado para revisão e gerenciamento de contratos, prever eventos jurídicos, possibilitando saber com antecedência o que se deve fazer em determinado caso e quais resultados serão obtidos automatização de casos usuais como casos de divórcio e uma segmento da advocacia que pode ajudar imensamente os profissionais da área é o uso da tecnologia para revisão de documentos e pesquisas jurídicas. Onde em um caso particular que o profissional de advocacia possa estar trabalhando, através do uso de Machine Learning possa conseguir rapidamente e facilmente casos semelhantes relevantes para o caso, onde para esse tipo de pesquisa onde o advogado teria que empregar horas e horas de trabalho repetitivo e exaustivo, buscando por padrões e detalhes para usar no caso atual. No qual maquinas são mais eficientes nessas revisões do que humanos e pode produzir um resultado baseado em estatística, sendo uma informação mais segura e válida. Isso acaba reduzindo as tarefas do advogado e esse tempo podendo ser usando em tarefas com mais prioridades e complexidade.

##### 1.1. Objetivos do Trabalho 
O objetivo geral deste trabalho é por meio da utilização de Machine Learning revisar documentos jurídicos de maneira eficaz.
Para a consecução deste objetivo foram estabelecidos os objetivos específicos:
•	Realizar uma investigação sobre o atual cenário jurídico em relação a tecnologia.
•	Propor uma solução tecnológica eficaz para revisão de documentos.
•	Estudar quais as melhores tecnologias para realizar o projeto. 

##### 1.2. Conteúdo do Trabalho
O presente trabalho está estruturado em seis Capítulos, cujo conteúdo é sucintamente apresentado a seguir: 
No Capítulo 1 é feita a introdução sobre o tema apresentado.
No Capítulo 2 é feita a fundamentação das tecnologias...
O Capítulo 3 apresenta o desenvolvimento da solução...

#### 2. FUNDAMENTAÇÃO TÉCNICA
Para a realização deste projeto será usado Machine Learning, utilizando a linguagem Python e as seguintes bibliotecas: NumPy, Pandas e Scikit-Learn.
##### 2.1. Machine Learning
Machine Learning tem seu aprendizado baseado em experiência. De modo que os computadores possam ser programados a partir dos dados que foram treinados anteriormente, adquirindo a habilidade de identificar elementos e suas características com alta probabilidade.

Estágios do Machine Learning:
•	Coleta de dados
•	Ordenação dos dados
•	Análise dos dados
•	Desenvolvimento do algoritmo 
•	Testar o algoritmo gerado 
•	Usar o algoritmo para gerar informações

Para identificar padrões, vários algoritmos são usados e podem ser divididos em dois grupos: Aprendizado supervisionado e aprendizado não supervisionado. 
O aprendizado supervisionado a máquina tem a habilidade de reconhecer elementos baseado nas amostras fornecidas. O computador estuda isso e desenvolve a habilidade de reconhecer novos dados.
O aprendizado não supervisionado a máquina recebe apenas o input de dados definido. A partir disso, a máquina poderá determinar a relação entre os dados inseridos e quaisquer outros dados hipotéticos. Diferentemente do aprendizado supervisionado, onde a máquina recebe alguns dados de verificação para aprendizado, o aprendizado não supervisionado implica que o próprio computador encontrará padrões e relacionamentos entre diferentes conjuntos de dados.
   
##### 2.2. Python
A linguagem Python será usada no desenvolvimento deste projeto devido a sua facilidade de uso e aprendizado, principalmente no âmbito de Machine Learning. Python é uma linguagem minimalista e intuitiva com frameworks completos para programar Machine Learning. 



##### 2.3. Biblioteca NumPy

NumPy é abreviação de Numerical Python, é a biblioteca mais universal e versátil para profissionais e iniciantes. Com esta ferramenta, é possível fazer operações com matrizes e matrizes multidimensionais com facilidade. Funções como álgebra linear e conversões numéricas também estão disponíveis.


##### 2.4. Pandas

Pandas é uma ferramenta de alto desempenho para apresentar quadros de dados. É possível carregar dados de praticamente qualquer fonte, calcular funções, criar parâmetros novos, criar consultas aos dados usando funções semelhantes ao SQL. Além disso, existem várias funções de transformação de matriz e outros métodos para obter informações de dados.

##### 2.5. Biblioteca Scikit-Learn

Scikit-Learn implementa uma ampla variedade de algoritmos de Machine Learning e facilita a conexão deles em aplicativos reais. É possível usar várias funções como regressão, clustering, seleção de modelos, pré-processamento, classificação e entre outros. 

#### 3. DESENVOLVIMENTO
O objetivo deste capítulo é apresentar o desenvolvimento de uma ferramenta capaz de gerar dados jurídicos em informação através do uso de Machine Learning utilizando a linguagem Python e frameworks.

##### 3.1. Arquitetura do Sistema
A arquitetura do sistema usada será o modelo cliente-servidor, usando aprendizado supervisionado. Onde a informação é solicitada por um cliente e processada em um servidor.  
##### 3.2. Dados necessários
Para que a máquina possa aprender a transformar dados em informação é necessário fornecer à máquina um volume de dados jurídicos para que possa haver aprendizado. O objetivo é utilizar tabelas no formato CSV ou similares contendo os dados necessários.
Esses dados devem estar formatados, sem erros ou alterações indesejadas para que possa ser lido e manipulado pelo aplicativo.
##### 3.3. Os passos a seguir usando Python
	     
Inicialmente o ambiente é preparado com a instalação dos frameworks necessários. Com o NumPy, Pandas e Scikit-Learn instalado corretamente já é possível começar a trabalhar com os dados.
Possuindo os dados já tratados adequadamente, é possível carregá-lo para o projeto. Deve ser feito a importação das bibliotecas necessárias possibilitando assim, começar a trabalhar com os dados, onde os dados poderão ser visualizados, separados em conjuntos e dimensões.

