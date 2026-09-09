---
description: Como abrir e rodar os notebooks do curso no Google Colab
---

# Antes de começar

Os notebooks do curso rodam no Google Colab, um Jupyter Notebook que vive
no navegador, sem instalar nada na sua máquina. Antes da primeira aula,
siga estes passos uma vez. Depois disso, o processo é sempre o mesmo.

{% hint style="warning" %}
**Erro comum:** abrir o notebook e já sair digitando, sem salvar uma cópia
primeiro. Se você fizer isso, corre o risco de perder tudo ao fechar a aba.
Sempre salve a cópia antes de escrever qualquer código.
{% endhint %}

## Passo a passo

1. Na página da aula, clique no botão **"Abrir no Colab"**, na seção
   Materiais. O notebook abre no seu navegador, em modo de leitura.
2. No menu, vá em **Arquivo → Salvar uma cópia no Drive**. Isso cria uma
   cópia sua, editável, guardada no seu Google Drive.
3. Feche a aba original (a de leitura) e continue só na cópia nova: ela
   abre numa aba separada.
4. Rode as células de cima para baixo. Para rodar uma célula, clique nela e
   aperte `Shift + Enter`, ou clique no botão de play que aparece à
   esquerda da célula.

## Se algo der errado

Se uma célula der um erro que não faz sentido — por exemplo, dizendo que
uma variável não existe, mesmo depois de você ter rodado a célula que a
cria — o ambiente provavelmente ficou com um estado antigo na memória.

Vá em **Ambiente de execução → Reiniciar sessão**, depois rode todas as
células de novo, começando pela primeira. Isso resolve a maioria dos
problemas.

## Onde ficam seus arquivos

O Colab salva sua cópia automaticamente no Google Drive, dentro da pasta
"Colab Notebooks". Você pode voltar a ela a qualquer momento pelo
[drive.google.com](https://drive.google.com).
