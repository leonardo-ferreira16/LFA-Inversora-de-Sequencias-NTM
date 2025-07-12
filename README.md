# Neural Turing Machine (NTM) - Inversora de Sequências de Bits

![Licença](https://img.shields.io/badge/licen%C3%A7a-MIT-blue.svg)
![Versão](https://img.shields.io/badge/vers%C3%A3o-1.0-brightgreen)

Uma ferramenta desenvolvida em Python com PyTorch, numpy e matplotlib para inverter sequências de 3 bits utilizando Máquinas de Turing com Redes Neurais, apresentando visualização de log dos testes e aprendizado da rede.

---

### Tabela de Conteúdos
* [Sobre o Projeto](#sobre-o-projeto)
* [Funcionalidades](#funcionalidades)
* [Screenshot da Aplicação](#screenshot-da-aplica%C3%A7%C3%A3o)
* [Tecnologias Utilizadas](#tecnologias-utilizadas)
* [Como Usar](#como-usar)
  * [Pré-requisitos](#pré-requisitos)
  * [Instalação](#instalação)
  * [Formato de Entrada do AFD](#formato-de-entrada-do-afd)
* [Estrutura do Projeto](#estrutura-do-projeto)
* [Reconhecimentos e Direitos Autorais](#reconhecimentos-e-direitos-autorais)
* [Licença](#licen%C3%A7a)

---

## Sobre o Projeto

Este projeto foi desenvolvido como parte da disciplina de Linguagens Formais e Autômatos. O objetivo é demonstrar uma implementação funcional e educacional de uma Neural Turing Machine, capaz de aprender operações simples de manipulação de memória, como a inversão de sequência.

A arquitetura implementa atenção por conteúdo, leitura e escrita em uma matriz de memória, com histórico de pesos para visualização.

## Funcionalidades

- **✅ Treinamento supervisionado para reversão de sequência binária:** Treina uma Neural Turing Machine (NTM) para aprender a reverter sequências binárias.
Utiliza memória externa diferenciável com leitura e escrita baseadas em atenção

- **✅ Módulo de memória com leitura e escrita diferenciáveis:** Visualiza o comportamento interno do modelo, incluindo pesos de leitura e escrita.
Permite entender como a NTM interage com a memória ao longo do tempo.

- **✅ Visualização de memória e pesos de leitura/escrita ao longo do tempo:** Gera mapas de calor dos acessos à memória em cada passo da sequência, facilitando a análise do processo de raciocínio da NTM.

- **✅ Teste interativo via terminal:** Permite ao usuário inserir manualmente sequências binárias e observar a saída gerada pela NTM em tempo real.

- **✅ Salvamento de modelo treinado:** Possibilita reutilizar o modelo após o treinamento, evitando retrabalho e acelerando experimentações futuras.

- **✅ Acurácia por bit e por sequência durante o treinamento:** Mede o desempenho detalhado da NTM, tanto em nível de bits quanto em sequências completas, exibindo gráficos ao longo das épocas.

## Screenshot da Aplicação

![alt text](screenshotdaaplicação.png)

## Tecnologias Utilizadas

- **Python 3**
- **PyTorch:** Biblioteca de machine learning que fornece os blocos básicos (tensores, autograd, módulos) usados para construir, treinar e avaliar a NTM.
- **NumPy:** Usado para conversões rápidas e manipulação de arrays fora do grafo computacional do PyTorch, especialmente para visualização de pesos.
- **Matplotlib:** Responsável pela geração de gráficos de perda, acurácia, pesos de leitura/escrita e visualização da matriz de memória.
- **tqdm:** Exibe barras de progresso elegantes durante o treinamento, facilitando o acompanhamento em tempo real das épocas

## Como Usar

Siga os passos abaixo para executar o projeto em sua máquina local.

### Pré-requisitos

- Python 3.8 ou superior;
- `pip` (gerenciador de pacotes do Python);
- GPU compatível com CUDA (opcional, mas recomendável);

### Instalação I (Executar em ambientes locais, como VScode)

1. **Clone o repositório:**
   ```sh
   git clone https://github.com/leonardo-ferreira16/LFA-Inversora-de-Sequencias-NTM
   cd [G3_INVERSORA_DE_SEQUÊNCIA_NTM]
   ```

2. **(Opcional, mas recomendado) Crie e ative um ambiente virtual:**
   ```sh
   # No Windows
   python -m venv venv
   .\venv\Scripts\activate

   # No macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instale as dependências necessárias:**
   ```sh
   pip install torch numpy matplotlib tqdm

   ```

4. **Execute a aplicação:**
   ```sh
   python ntm_p3.py  
   ```

 **Instalação II (Execução no Ambiente do Google Collab):** 
   [Utilizar o arquivo NTM_P3.ipynb e upar no ambiente ou acessar o seguinte link:](https://colab.research.google.com/drive/1RkPzw5ITty0ta1F05Fay0TGxcsThxbKt?usp=sharing)
 

### Formato de Entrada para Testes com a NTM

Na parte interativa do notebook, a NTM espera sequências binárias fixas com comprimento definido pelo parâmetro SEQ_LEN (por padrão, SEQ_LEN = 3).

- **Formato da Entrada:** Digite diretamente no terminal ou input do notebook uma sequência de 0s e 1s, com o número exato de bits especificado por SEQ_LEN.
- **Formato da Saída Esperada::** A NTM tentará inverter a sequência binária, retornando a sequência na ordem oposta.

**Exemplo de definição completa:**

```
Sua sequência binária (3 dígitos): 101
  Entrada: [1.0, 0.0, 1.0]
  Saída Esperada: [1.0, 0.0, 1.0]
  Saída Predita: [1.0, 0.0, 1.0]
  Correto: Sim

```

## Estrutura do Projeto

O código possui blocos para separar responsabilidades:

- `Bloco: Definição da Classe NTM`: Contém a implementação completa da Neural Turing Machine, incluindo controlador LSTM, mecanismo de leitura/escrita, memória externa e histórico de pesos;
- `Bloco: Geração do Dataset`: Define a função create_dataset() para gerar pares de entrada/saída com sequências binárias e suas inversões;
- `Bloco: Treinamento da NTM`: Configura hiperparâmetros, realiza o treinamento com otimizador RMSprop, clip de gradiente, e registra métricas de desempenho (loss, acurácia por bit e por sequência);
- `Bloco: Visualização dos Resultados`: Gera gráficos com as curvas de perda e acurácia ao longo das épocas de treinamento;
- `Bloco: Testes Fixos com Saída Detalhada`: Avalia a NTM em todas as possíveis entradas de 3 bits, mostrando saída esperada, predição, e visualização dos pesos de leitura/escrita para alguns exemplos;
- `Bloco: Teste Interativo via Terminal`: Permite ao usuário digitar sequências binárias manualmente e observar a saída da NTM em tempo real, com validação de acerto;
- ` Bloco: Visualização da Memória e Pesos`: Exibe visualmente a matriz de memória da NTM e o comportamento dos pesos de leitura e escrita ao longo do tempo para entradas específicas;
---

## Reconhecimentos e Direitos Autorais

* **@autor:** Leonardo Abreu Ferreira, Pedro Arthur Da Silva Guimarães
* **@contato:** leonardo.abreu@discente.ufma.br, arthurguimaraespds@gmail.com
* **@data última versão:** 10 de julho de 2025
* **@versão:** 1.0
* **@outros repositórios:** [text](https://github.com/Pedrokodart1), (https://github.com/leonardo-ferreira16)
* **@Agradecimentos:** Universidade Federal do Maranhão (UFMA), Professor Doutor Thales Levi Azevedo Valente, e colegas de curso.

## Licença

Este material é resultado de um trabalho acadêmico para a disciplina LINGUAGENS FORMAIS E AUTÔMATOS, sob a orientação do professor Dr. THALES LEVI AZEVEDO VALENTE, semestre letivo 2025.1, curso Engenharia da Computação, na Universidade Federal do Maranhão (UFMA). Todo o material sob esta licença é software livre: pode ser usado para fins acadêmicos e comerciais sem nenhum custo. Não há papelada, nem royalties, nem restrições de "copyleft" do tipo GNU. Ele é licenciado sob os termos da Licença MIT, conforme descrito abaixo, e, portanto, é compatível com a GPL e também se qualifica como software de código aberto. É de domínio público. Os detalhes legais estão abaixo. O espírito desta licença é que você é livre para usar este material para qualquer finalidade, sem nenhum custo. O único requisito é que, se você usá-los, nos dê crédito.

Licenciado sob a Licença MIT. Para mais informações: [https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT)

> Copyright (c) 2025 Leonardo Abreu Ferreira , Pedro Arthur Da Silva Guimarães
>
> Permissão é concedida, gratuitamente, a qualquer pessoa que obtenha uma cópia deste software e dos arquivos de documentação associados (o "Software"), para lidar no Software sem restrição, incluindo sem limitação os direitos de usar, copiar, modificar, mesclar, publicar, distribuir, sublicenciar e/ou vender cópias do Software, e permitir pessoas a quem o Software é fornecido a fazê-lo, sujeito às seguintes condições:
>
> Este aviso de direitos autorais e este aviso de permissão devem ser incluídos em todas as cópias ou partes substanciais do Software.
>
> O SOFTWARE É FORNECIDO "COMO ESTÁ", SEM GARANTIA DE QUALQUER TIPO, EXPRESSA OU IMPLÍCITA, INCLUINDO MAS NÃO SE LIMITANDO ÀS GARANTIAS DE COMERCIALIZAÇÃO, ADEQUAÇÃO A UM DETERMINADO FIM E NÃO INFRINGÊNCIA. EM NENHUM CASO OS AUTORES OU DETENTORES DE DIREITOS AUTORAIS SERÃO RESPONSÁVEIS POR QUALQUER RECLAMAÇÃO, DANOS OU OUTRA RESPONSABILIDADE, SEJA EM AÇÃO DE CONTRATO, TORT OU OUTRA FORMA, DECORRENTE DE, FORA DE OU EM CONEXÃO COM O SOFTWARE OU O USO OU OUTRAS NEGOCIAÇÕES NO SOFTWARE.