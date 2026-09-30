import re
 
# Stopwords em português (Lista gerada por IA) 
STOPWORDS_PT = {
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com",
    "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como",
    "mas", "ao", "ele", "das", "à", "seu", "sua", "ou", "quando", "muito",
    "nos", "já", "eu", "também", "só", "pelo", "pela", "até", "isso",
    "ela", "entre", "depois", "sem", "mesmo", "aos", "ter", "seus",
    "quem", "nas", "me", "esse", "eles", "estão", "você", "tinha", "foram",
    "essa", "num", "nem", "suas", "meu", "às", "minha", "têm", "numa",
    "pelos", "elas", "havia", "seja", "qual", "será", "nós", "tenho",
    "lhe", "deles", "essas", "esses", "pelas", "este", "dele", "tu",
    "te", "vocês", "vos", "lhes", "meus", "minhas", "teu", "tua",
    "teus", "tuas", "nosso", "nossa", "nossos", "nossas", "aquele",
    "aquela", "aqueles", "aquelas", "isto", "aquilo", "estou", "está",
    "estamos", "estão", "estive", "esteve", "estivemos", "estiveram",
    "estava", "estávamos", "estavam", "estivera", "estivéramos",
    "esteja", "estejamos", "estejam", "estivesse", "estivéssemos",
    "estivessem", "estiver", "estivermos", "estiverem", "hei", "há",
    "havemos", "hão", "houve", "houvemos", "houveram", "houvera",
    "houvéramos", "haja", "hajamos", "hajam", "houvesse", "houvéssemos",
    "houvessem", "houver", "houvermos", "houverem", "houverei", "houverá",
    "houveremos", "houverão", "houveria", "houveríamos", "houveriam",
    "sou", "somos", "são", "era", "éramos", "eram", "fui", "foi",
    "fomos", "foram", "fora", "fôramos", "seja", "sejamos", "sejam",
    "fosse", "fôssemos", "fossem", "for", "formos", "forem", "serei",
    "será", "seremos", "serão", "seria", "seríamos", "seriam", "tenho",
    "tem", "temos", "têm", "tinha", "tínhamos", "tinham", "tive",
    "teve", "tivemos", "tiveram", "tivera", "tivéramos", "tenha",
    "tenhamos", "tenham", "tivesse", "tivéssemos", "tivessem",
    "tiver", "tivermos", "tiverem", "terei", "terá", "teremos",
    "terão", "teria", "teríamos", "teriam", "is", "are", "the",
    "todo", "toda", "todos", "todas", "outro", "outra", "outros",
    "outras", "num", "nuns", "numa", "numas", "aqui", "ali", "lá",
    "cá", "não", "sim", "talvez", "isso", "isto", "aquilo", "tudo",
    "nada", "alguém", "ninguém", "cada", "qualquer", "quaisquer",
    "tanto", "tanta", "tantos", "tantas", "quanto", "quanta",
    "poucos", "poucas", "pouco", "pouca", "muito", "muita", "muitos",
    "muitas", "mais", "menos", "bem", "mal", "melhor", "pior",
    "ainda", "então", "porém", "contudo", "todavia", "entretanto",
    "portanto", "logo", "pois", "porque", "embora", "caso", "desde",
    "enquanto", "durante", "antes", "depois", "sobre", "sob", "entre",
    "contra", "perante", "mediante", "conforme", "segundo", "através",
}
 
 
def calcular_riqueza_lexical(texto):
    palavras_originais = texto.split()
    total_palavras_original = len(palavras_originais)
 
    if total_palavras_original == 0:
        return {
            "riqueza_lexical": 0,
            "total_original": 0,
            "total_unicas_filtradas": 0,
        }
 
    texto_min = texto.lower()
    texto_limpo = re.sub(r'[^a-záàâãéèêíïóôõöúç\s]', '', texto_min)
 
    palavras_filtradas = [p for p in texto_limpo.split() if p not in STOPWORDS_PT and p]
    palavras_unicas = set(palavras_filtradas)
    total_unicas = len(palavras_unicas)
 
    riqueza = total_unicas / total_palavras_original
 
    return {
        "riqueza_lexical": riqueza,
        "total_original": total_palavras_original,
        "total_unicas_filtradas": total_unicas,
    }
 
 
def processar_lista(perguntas: list[str]) -> str:
    linhas = ["Índice\tRiqueza Lexical"]
 
    for i, pergunta in enumerate(perguntas, start=1):
        resultado = calcular_riqueza_lexical(pergunta)
        riqueza   = f"{resultado['riqueza_lexical']:.4f}".replace(".", ",")  
        linhas.append(f"{i}\t{riqueza}")
 
    return "\n".join(linhas)
 
 
if __name__ == "__main__":
    respostas_consumidor_v2 = [
        #adicione as respostas geradas aqui, exemplo: usar respotas em results/
    ]
    
    texto_formatado = processar_lista(respostas_consumidor_v2)
    
    nome_arquivo = "resultados_riqueza_lexical.txt"
    with open(nome_arquivo, "w", encoding="utf-8") as arquivo:
        arquivo.write(texto_formatado)
        
    print(f"O arquivo '{nome_arquivo}' foi criado com sucesso!")