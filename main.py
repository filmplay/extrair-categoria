import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import csv
import io
from dotenv import load_dotenv
from groq import Groq
from dateutil import parser as dateparser

# Carregar variáveis de ambiente
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("Chave de API da Groq não encontrada. Defina GROQ_API_KEY no .env")

# Inicializar cliente Groq
client = Groq(api_key=groq_api_key)

app = FastAPI()

DEFAULT_CATEGORIES: List[str] = [
    "Alimentação", "Transporte", "Moradia", "Entretenimento",
    "Utilidades", "Saúde", "Educação", "Renda", "Transferência",
    "Contas", "Outros"
]

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Alimentação": [
        'salário', 'supermercado', 'mercado', 'padaria', 'ataca', 'atacadão',
        'assai', 'hortifruti', 'açougue', 'Savegnago'
    ],
    "Transporte": [
        'posto', 'combustível', 'uber', '99', 'estacionamento', 'pedágio'
    ],
    "Moradia": [
        'aluguel', 'condomínio', 'imobiliária'
    ],
    "Utilidades": [
        'luz', 'água', 'gás', 'energia', 'sanepar', 'copel'
    ],
    "Saúde": [
        'drogaria', 'farmácia', 'hospital', 'farma', 'clínica', 'laboratório', 'plano de saúde'
    ],
    "Educação": [
        'escola', 'universidade', 'curso'
    ],
    "Contas": [
        'tarifa', 'boleto', 'fatura', 'pagamento'
    ],
    "Entretenimento": [
        'netflix', 'cinema', 'restaurante', 'parque', 'mc donalds'
    ],
    "Investimentos": [
        'renda fixa', 'ações', 'resgate'
    ],
    "Doações": [
        'igreja', 'ong', 'contribuição'
    ],
    "Renda": [
        'recebimento', 'rendimento'
    ],
    "Transferência": [
        'pix', 'transferência', 'enviada'
    ]
}

class TransacaoCategorizada(BaseModel):
    descricao: str
    valor: float
    tipo: str
    categoria: str
    data_transacao: str
    id_transacao: Optional[str]
    portador: str

def _formatar_data(data: str) -> str:
    """
    Converte qualquer data para o formato YYYY/MM/DD
    """
    try:
        dt = dateparser.parse(data, dayfirst=True, fuzzy=True)
        return dt.strftime("%Y/%m/%d")
    except Exception:
        return data  # Se falhar, mantém a original


def classificar_transacao(descricao: str, tipo: str) -> str:
    prompt = (
        "Você é um assistente especializado em finanças. "
        "Classifique a seguinte transação bancária em uma das categorias: "
        f"{', '.join(CATEGORY_KEYWORDS.keys())}, Outros.\n"
        f"Descrição da transação: '{descricao}'\n"
        f"Tipo da transação: '{tipo}'\n"
        "Responda apenas com o nome da categoria."
    )
    try:
        resposta = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Você é um assistente de categorização financeira."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=5
        )
        categoria = resposta.choices[0].message.content.strip()

        # Ajuste para transferências
        if categoria.lower() == "transferência":
            return "Transferência recebida" if tipo.lower() == "entrada" else "Transferência enviada"

        return categoria if categoria else "Outros"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na Groq API: {e}")

@app.post(
    "/categorizar",
    summary="Classifica transações via CSV ou TXT (Nubank, Itaú e outros) e retorna CSV"
)
async def categorizar(
    file: UploadFile = File(...),
    banco: str = Form(...),
    portador: str = Form(...),
    config: Optional[str] = Form(None)
):
    raw = await file.read()
    text = raw.decode('utf-8', errors='ignore')
    await file.close()

    resultado: List[TransacaoCategorizada] = []

    if banco.lower() == 'itau':
        for ln, line in enumerate(text.splitlines(), start=1):
            parts = [p.strip() for p in line.split(';')]
            if len(parts) != 3:
                continue
            data, descricao, raw_valor = parts
            try:
                valor = float(raw_valor.replace(',', '.'))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Valor inválido na linha {ln}")

            tipo = "Entrada" if valor > 0 else "Saída"
            categoria = classificar_transacao(descricao, tipo)
            data_transacao = _formatar_data(data)
            resultado.append(TransacaoCategorizada(
                descricao=descricao,
                valor=valor,
                tipo=tipo,
                categoria=categoria,
                data_transacao=data_transacao,
                id_transacao=None,
                portador=portador
            ))
    else:  # Nubank ou outros CSV
        reader = csv.DictReader(io.StringIO(text))
        required = {'Data', 'Descrição', 'Valor', 'Identificador'}
        if not required.issubset(reader.fieldnames or []):
            faltando = required - set(reader.fieldnames or [])
            raise HTTPException(status_code=400, detail=f"Colunas faltando: {', '.join(faltando)}")

        for row in reader:
            try:
                valor = float(row['Valor'].replace(',', '.'))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Valor inválido na linha {reader.line_num}")

            tipo = "Entrada" if valor > 0 else "Saída"
            categoria = classificar_transacao(row['Descrição'], tipo)
            data_transacao = _formatar_data(row['Data'])
            resultado.append(TransacaoCategorizada(
                descricao=row['Descrição'],
                valor=valor,
                tipo=tipo,
                categoria=categoria,
                data_transacao=data_transacao,
                id_transacao=row.get('Identificador'),
                portador=portador
            ))

    # Gera CSV de saída
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=resultado[0].dict().keys())
    writer.writeheader()
    for item in resultado:
        writer.writerow(item.dict())

    output.seek(0)
    return StreamingResponse(output, media_type="text/csv", headers={
        "Content-Disposition": f"attachment; filename=categorias_{banco}.csv"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)