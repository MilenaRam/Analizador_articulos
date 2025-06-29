
"""
@author: Milena Ramírez
"""
import os
import re
import nltk
import docx
import PyPDF2
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Descargar recursos de NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class ArticleComparator:
    def __init__(self):
# Stopwords científicas personalizadas
        self.scientific_stopwords = set([
            'study', 'research', 'analysis', 'paper', 'article', 'method', 'result', 'conclusion', 'discuss', 'present', 'show', 'find',
              'investigate', 'examine', 'evaluate', 'assess', 'determine', 'approach', 'technique', 'procedure', 'process', 'system'
        ])

# Conectores comunes a excluir
        self.connectors = set([
            # Español
            'además', 'sin embargo', 'aunque', 'por lo tanto', 'por consiguiente', 'por ejemplo', 'en cambio', 'en resumen', 'por otro lado', 'a pesar de',           
            'con respecto a', 'debido a', 'asimismo', 'finalmente', 'entonces', 'luego', 'después', 'antes', 'mientras', 'porque', 'ya que', 'cuando',          
            'donde', 'cómo', 'para', 'pero', 'y', 'o', 'ni', 'que', 'como', 'si',
            # Inglés
            'however', 'therefore', 'thus', 'for example', 'on the other hand', 'in contrast', 'moreover', 'furthermore', 'because', 'although',           
            'since', 'when', 'where', 'while', 'but', 'and', 'or', 'if', 'so', 'as'
        ])

        self.section_keywords = {
            'abstract': ['abstract', 'summary'],
            'introduction': ['introduction', 'background'],
            'methodology': ['method', 'methodology', 'approach', 'procedure'],
            'results': ['result', 'finding', 'outcome'],
            'discussion': ['discussion', 'conclusion', 'implication'],
            'references': ['reference', 'bibliography', 'citation']
        }
# Cargar archivo según el formato
    def cargar_articulo(self, ruta_archivo):
# Verifica si el archivo existe, si no lanza error
        if not os.path.exists(ruta_archivo):
            raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")
# Extrae la extensión del archivo
        extension = os.path.splitext(ruta_archivo)[1].lower()
# Llama al método correspondiente según la extensión  
        if extension == '.txt':
            return self._cargar_txt(ruta_archivo)
        elif extension == '.pdf':
            return self._cargar_pdf(ruta_archivo)
        elif extension == '.docx':
            return self._cargar_docx(ruta_archivo)
        else:
            raise ValueError(f"Formato no soportado: {extension}")
            
# Carga contenido de archivo .txt
    def _cargar_txt(self, ruta):
        with open(ruta, 'r', encoding='utf-8') as file:
            return file.read()
        
# Carga contenido de archivo .pdf usando PyPDF2
    def _cargar_pdf(self, ruta):
        texto = ""
        try:
            with open(ruta, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    texto += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error al leer PDF: {e}")
            return ""
        return texto

# Carga contenido de archivo .docx usando python-docx
    def _cargar_docx(self, ruta):
        try:
            doc = docx.Document(ruta)
            texto = ""
            for paragraph in doc.paragraphs:
                texto += paragraph.text + "\n"
        except Exception as e:
            print(f"Error al leer DOCX: {e}")
            return ""
        return texto
    
# Preprocesamiento de texto: limpieza, minúsculas, eliminar conectores
    def preprocesar_texto(self, texto):
        texto = re.sub(r'\s+', ' ', texto)
        texto = re.sub(r'[^\w\s\.]', ' ', texto)
        texto = texto.lower().strip()
        for connector in self.connectors:
            texto = re.sub(r'\b' + re.escape(connector) + r'\b', '', texto)
        return texto

# Extrae información estructural del texto
    def extraer_estructura(self, texto):
        lineas = texto.split('\n')
        parrafos = [p for p in texto.split('\n\n') if len(p.strip()) > 20]
        palabras = word_tokenize(texto)
        oraciones = sent_tokenize(texto)
        secciones_encontradas = {}
        for seccion, keywords in self.section_keywords.items():
            encontrada = any(keyword in texto.lower() for keyword in keywords)
            secciones_encontradas[seccion] = encontrada
        referencias = len(re.findall(r'\[\d+\]|\(\d{4}\)|et al\.', texto))
        figuras = len(re.findall(r'fig\.|figure|tabla|table', texto, re.IGNORECASE))
        estructura = {
            'num_palabras': len(palabras),
            'num_oraciones': len(oraciones),
            'num_parrafos': len(parrafos),
            'num_referencias': referencias,
            'num_figuras_tablas': figuras,
            'secciones': secciones_encontradas,
            'palabras_por_oracion': len(palabras) / len(oraciones) if oraciones else 0,
            'oraciones_por_parrafo': len(oraciones) / len(parrafos) if parrafos else 0
        }
        return estructura
    
# Extrae palabras clave mediante TF-IDF    
    def extraer_palabras_clave(self, texto, n_palabras=20):
        palabras = word_tokenize(texto.lower())
        stop_words = set(stopwords.words('english') + stopwords.words('spanish'))
        stop_words.update(self.scientific_stopwords)
        stop_words.update(self.connectors)
        palabras_filtradas = [
            palabra for palabra in palabras
            if (len(palabra) >= 3 and
                palabra.isalpha() and
                palabra not in stop_words)
        ]
        contador = Counter(palabras_filtradas)
        vectorizer = TfidfVectorizer(max_features=100, stop_words=stop_words)
        try:
            tfidf_matrix = vectorizer.fit_transform([texto])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            palabras_clave = []
            for palabra, score in zip(feature_names, tfidf_scores):
                if score > 0:
                    palabras_clave.append((palabra, score))
            palabras_clave.sort(key=lambda x: x[1], reverse=True)
            return [palabra for palabra, _ in palabras_clave[:n_palabras]]
        except:
            return [palabra for palabra, _ in contador.most_common(n_palabras)]
        
# Calcula similitud Jaccard entre dos conjuntos
    def calcular_similitud_jaccard(self, set1, set2):
        interseccion = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return interseccion / union if union > 0 else 0
    
# Compara estructuras numéricas y secciones
    def comparar_estructura(self, struct1, struct2):
        similitudes = []
        campos_numericos = ['num_palabras', 'num_oraciones', 'num_parrafos',
                            'palabras_por_oracion', 'oraciones_por_parrafo']
        for campo in campos_numericos:
            val1, val2 = struct1[campo], struct2[campo]
            if val1 > 0 and val2 > 0:
                similitud = min(val1, val2) / max(val1, val2)
                similitudes.append(similitud)
        secciones1 = set(k for k, v in struct1['secciones'].items() if v)
        secciones2 = set(k for k, v in struct2['secciones'].items() if v)
        sim_secciones = self.calcular_similitud_jaccard(secciones1, secciones2)
        similitudes.append(sim_secciones)
        return sum(similitudes) / len(similitudes) if similitudes else 0
    
# Compara dos artículos: carga, preprocesa, analiza, calcula similitudes
    def comparar_articulos(self, ruta1, ruta2):
        print("Cargando artículos...")
        texto1 = self.cargar_articulo(ruta1)
        texto2 = self.cargar_articulo(ruta2)
        if not texto1 or not texto2:
            print("Error: No se pudieron cargar los artículos")
            return None
        print("Preprocesando textos...")
        texto1_limpio = self.preprocesar_texto(texto1)
        texto2_limpio = self.preprocesar_texto(texto2)
        print("Extrayendo características...")
        struct1 = self.extraer_estructura(texto1_limpio)
        struct2 = self.extraer_estructura(texto2_limpio)
        keywords1 = self.extraer_palabras_clave(texto1_limpio)
        keywords2 = self.extraer_palabras_clave(texto2_limpio)
        print("Calculando similitudes...")
        sim_keywords = self.calcular_similitud_jaccard(set(keywords1), set(keywords2))
        sim_estructura = self.comparar_estructura(struct1, struct2)
        all_stopwords = set(stopwords.words('english') + stopwords.words('spanish'))
        all_stopwords.update(self.scientific_stopwords)
        all_stopwords.update(self.connectors)
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=all_stopwords)
        try:
            tfidf_matrix = vectorizer.fit_transform([texto1_limpio, texto2_limpio])
            sim_contenido = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            sim_contenido = 0
        keywords_comunes = list(set(keywords1).intersection(set(keywords2)))
        resultado = {
            'similitud_keywords': sim_keywords,
            'similitud_estructura': sim_estructura,
            'similitud_contenido': sim_contenido,
            'keywords_comunes': keywords_comunes,
            'estructura1': struct1,
            'estructura2': struct2,
            'keywords1': keywords1[:10],
            'keywords2': keywords2[:10]
        }
        return resultado
    
# Genera un reporte en formato de texto legible
    def generar_reporte(self, resultado, archivo1, archivo2):
        if not resultado:
            return "No se pudo generar el reporte"
 # Reporte con estructura y similitudes
        reporte = f"""
{'='*60}
          COMPARACIÓN DE ARTÍCULOS CIENTÍFICOS
{'='*60}

ARCHIVOS COMPARADOS:
- Artículo 1: {os.path.basename(archivo1)}
- Artículo 2: {os.path.basename(archivo2)}

{'='*60}
SIMILITUDES GENERALES:
{'='*60}
• Similitud de palabras clave:    {resultado['similitud_keywords']:.1%}
• Similitud estructural:          {resultado['similitud_estructura']:.1%}  
• Similitud de contenido:         {resultado['similitud_contenido']:.1%}

{'='*60}
ANÁLISIS ESTRUCTURAL:
{'='*60}
                        Artículo 1    Artículo 2    Ratio
Palabras:              {resultado['estructura1']['num_palabras']:>10}    {resultado['estructura2']['num_palabras']:>10}    {resultado['estructura1']['num_palabras']/resultado['estructura2']['num_palabras']:.2f}
Oraciones:             {resultado['estructura1']['num_oraciones']:>10}    {resultado['estructura2']['num_oraciones']:>10}    {resultado['estructura1']['num_oraciones']/resultado['estructura2']['num_oraciones']:.2f}
Párrafos:              {resultado['estructura1']['num_parrafos']:>10}    {resultado['estructura2']['num_parrafos']:>10}    {resultado['estructura1']['num_parrafos']/resultado['estructura2']['num_parrafos']:.2f}
Referencias:           {resultado['estructura1']['num_referencias']:>10}    {resultado['estructura2']['num_referencias']:>10}    {(resultado['estructura1']['num_referencias']/max(resultado['estructura2']['num_referencias'],1)):.2f}

{'='*60}
PALABRAS CLAVE:
{'='*60}
Palabras clave comunes ({len(resultado['keywords_comunes'])}):
{', '.join(resultado['keywords_comunes'][:15]) if resultado['keywords_comunes'] else 'Ninguna'}

Top palabras clave - Artículo 1:
{', '.join(resultado['keywords1'])}

Top palabras clave - Artículo 2:
{', '.join(resultado['keywords2'])}

{'='*60}
CONCLUSIÓN:
{'='*60}"""
        sim_promedio = (resultado['similitud_keywords'] +
                        resultado['similitud_estructura'] +
                        resultado['similitud_contenido']) / 3
        if sim_promedio >= 0.7:
            conclusion = "Los artículos son MUY SIMILARES"
        elif sim_promedio >= 0.5:
            conclusion = "Los artículos son MODERADAMENTE SIMILARES"
        elif sim_promedio >= 0.3:
            conclusion = "Los artículos tienen ALGUNAS SIMILITUDES"
        else:
            conclusion = "Los artículos son DIFERENTES"
        reporte += f"\n{conclusion} (Similitud promedio: {sim_promedio:.1%})\n"
        reporte += "="*60 + "\n"
        return reporte

# Punto de entrada de la aplicación por consola
def main():
    print("=== COMPARADOR DE ARTÍCULOS CIENTÍFICOS ===\n")
    archivo1 = input("Ingresa aquí la ruta del primer artículo: ").strip().strip('"')
    archivo2 = input("Ingresa aquí la ruta del segundo artículo: ").strip().strip('"')
    try:
        comparador = ArticleComparator()
        resultado = comparador.comparar_articulos(archivo1, archivo2)
        if resultado:
            reporte = comparador.generar_reporte(resultado, archivo1, archivo2)
            print(reporte)
            guardar = input("\n¿Guardar reporte en archivo? (s/n): ").lower()
            if guardar == 's':
                nombre_reporte = f"reporte_comparacion_{os.path.basename(archivo1)}_vs_{os.path.basename(archivo2)}.txt"
                with open(nombre_reporte, 'w', encoding='utf-8') as f:
                    f.write(reporte)
                print(f"Reporte guardado como: {nombre_reporte}")
    except Exception as e:
        print(f"Error durante la comparación: {e}")


if __name__ == "__main__":
    main()
