import pandas as pd
import tiktoken
import openai


from errors import InexistantChat, VoidChatHistory, NoClientMessages, S3UploadError
from connection import messages_db, chats_db, openai_key
from bson.objectid import ObjectId
from typing import List, Dict
from collections import defaultdict


def num_tokens_from_messages(messages: List[Dict], model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def split_dataframe(df: pd.DataFrame, n_parts: int) -> Dict[int, pd.DataFrame]:
    """
    Split a chat dataframe into n parts.
    """
    dataframe_dict = defaultdict()
    n_rows = len(df)
    n_rows_per_part = n_rows // n_parts
    n_rows_leftover = n_rows % n_parts
    start = 0
    end = n_rows_per_part
    for i in range(n_parts):
        if i == n_parts - 1:
            end += n_rows_leftover
        dataframe_dict[i] = df.iloc[start:end]
        start = end
        end += n_rows_per_part
    return dataframe_dict


def generate_gpt_message_input(
    messages_df: pd.DataFrame, input_type: str = "full"
) -> List[Dict]:
    """ """
    initial_prompt = 'Eu vou fornecer uma conversa entre um cliente(C) e um atendente(A) e você deve me retornar o sentimento do cliente em,\
 no máximo 2 palavras. Por exemplo: "Muito Satisfeito" ou " Insatisfeito".\n \
\nOs sentimentos possíveis do cliente, e consequentemente as opções de classificação de sentimento, são: "Muito Satisfeito", "Satisfeito", "Neutro", "Insatisfeito", "Muito Insatisfeito".'
    single_message = ""
    messages = []
    messages.append({"role": "system", "content": f"{initial_prompt}"})

    if input_type == "client":
        single_message = format_client_messages_to_gpt(messages_df)["string"]
        messages.append({"role": "user", "content": single_message})
    if input_type == "full":
        single_message = format_chat_to_gpt(messages_df)["string"]
        messages.append({"role": "user", "content": single_message})

    return messages


def get_single_chat_gpt_sentiment(gpt_formatted_messages):
    """
    Get sentiment of a single chat using GPT-3
    """
    # Get sentiment of the chat
    openai.api_key = openai_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=gpt_formatted_messages,
        temperature=0,
    )
    return response["choices"][0]["message"]["content"]


def get_chat_split(msg_df: pd.DataFrame):
    # Formata o chat para o formato de entrada da OpenAI API
    initial_input_messages = generate_gpt_message_input(msg_df)

    # Calcula o número de tokens do chat
    n_tokens = num_tokens_from_messages(initial_input_messages)

    # Divisão da dataframe baseado nos tokens
    chat_parts = 1

    dataframe_dict = split_dataframe(msg_df, chat_parts)

    # Find minimum division of dataframe for number of tokens smaller than 3k
    while n_tokens > 3000:
        major_n_tokens = 0
        dataframe_dict = split_dataframe(msg_df, chat_parts)

        # Get dataframe with biggest token number and update n_tokens
        for key, df in dataframe_dict.items():
            gpt_input_messages = generate_gpt_message_input(df)
            n_tokens_local = num_tokens_from_messages(gpt_input_messages)
            if n_tokens_local > major_n_tokens:
                major_n_tokens = n_tokens_local
                n_tokens = n_tokens_local

        chat_parts += 1

    return dataframe_dict


def sentiment_to_score(sentiment: str):
    score = 0

    if sentiment == "Satisfeito":
        score = 1
    elif sentiment == "Muito Satisfeito":
        score = 2
    elif sentiment == "Neutro":
        score = 0
    elif sentiment == "Muito Insatisfeito":
        score = -2
    elif sentiment == "Insatisfeito":
        score = -1
    return score


def get_gpt_sentiment(messages_df: pd.DataFrame):
    """
    Get the sentiment of a chat in the format of a list of GPT3 messages.

    Input:
        messages_df: Pandas DataFrame, the chat history.

    Output:
        sentiment_output: Dictionay containing "average"
        (average sentiment scores of all chat parts) and "sentiment_dict"
        (dictonary of scores of each chat part).
    """

    sentiment = 0
    sentiments = defaultdict(lambda: None)

    chat_parts = get_chat_split(messages_df)
    print(f"The chat was divided into {len(chat_parts)} part(s)")
    for order, chat_df in chat_parts.items():
        gpt_formatted_messages = generate_gpt_message_input(chat_df)
        str_sentiment = get_single_chat_gpt_sentiment(gpt_formatted_messages)
        partial_sentiment = sentiment_to_score(str_sentiment)
        sentiment += partial_sentiment
        sentiments[order] = partial_sentiment

    sentiment /= len(chat_parts)

    sentiment_output = {"average": sentiment, "sentiment_dict": sentiments}

    return sentiment_output


def chat_id_verification(chat_id: str):
    chat_check = chats_db.find_one({"_id": ObjectId(chat_id)})

    if chat_check == None:
        raise InexistantChat(f"Erro: Arquivo do chat ${chat_id} não existe.")


def get_chat_messages(chat_id: str):
    messages = []
    query = {
        "chat": ObjectId(chat_id),
        "type": "chat",
        "text": {"$exists": "true"},
    }

    messages = messages_db.find(query).sort("send_date", 1)
    messages = list(messages)

    if len(messages) < 3:
        raise VoidChatHistory(
            "Chat não tem mensagens suficientes para uma análise de sentimento"
        )

    return messages


def import_data(messages: list) -> pd.DataFrame:
    order = 1

    messages_dict = {
        "id": [],
        "text": [],
        "source": [],
        "send_date": [],
        "order_in_chat": [],
    }

    for m in messages:
        # Get message id
        messages_dict["id"].append(m["_id"])

        # Get message source
        if m["is_out"]:
            messages_dict["source"].append("A")
            # Add order in chat sequence for Attendant:
            messages_dict["order_in_chat"].append("NA")
        else:
            messages_dict["source"].append("C")
            # Add order in chat sequence for Client:
            messages_dict["order_in_chat"].append(order)
            order += 1

        # Get message text
        messages_dict["text"].append(m["text"])

        # Get message datetime
        messages_dict["send_date"].append(m["timestamp"])

    messages_df = pd.DataFrame(data=messages_dict)

    # Sort messages by send_date
    messages_df.sort_values(by=["send_date"], inplace=True)

    return messages_df


def message_cleanup(messages_df):
    cleaned_messages_df = messages_df[messages_df["source"] == "C"]
    cleaned_messages_df.reset_index(drop=True, inplace=True)

    return cleaned_messages_df


# Function to calculate individual message weight:
def calculate_weight(order: int, n_messages: int):
    if n_messages < 1:
        raise NoClientMessages("Não há mensagens de clientes no chat")
    den = 0
    for i in range(1, n_messages + 1):
        den += i**2
    w = (order**2) / den

    return w


# Function to generate a dataframe with weighted messages:
def generate_weighted_df(df: pd.DataFrame):
    n_messages = df.shape[0]
    df = df.assign(
        message_weight=df.apply(
            lambda x: calculate_weight(x["order_in_chat"], n_messages), axis=1
        )
    )

    return df


# Function to calculate chat sentiment based on message weights and classification score
def calculate_chat_sentiment_coef(df):
    num = 0
    den = 0
    for _, row in df.iterrows():
        num += row["classification_label"] * row["message_weight"]
        den += row["message_weight"]
    coef = num / den
    return coef


# Function to generate the satisfaction label of the chat:
def generate_sentiment_label(coef: float):
    label = ""
    if coef > 2 or coef < -2:
        return "Houve um erro, por favor entre em contato com o suporte da ChatGuru"

    if coef <= -1:
        label = "Muito Insatisfeito"
    elif coef < -0.2:
        label = "Insatisfeito"
    elif coef <= 0.2:
        label = "Neutro"
    elif coef < 1:
        label = "Satisfeito"
    else:
        label = "Muito Satisfeito"

    return label


def format_chat(messages: list):
    chat_text = []

    for m in messages:
        # Get message source
        if m["is_out"]:
            attendant_message = f"[Atendente] ({m['timestamp']}) {m['text']}"
            chat_text.append(attendant_message)
        else:
            client_message = f"[Cliente] ({m['timestamp']}) {m['text']}"
            chat_text.append(client_message)

    return chat_text


def format_chat_to_gpt(messages_df: pd.DataFrame):
    chat_text = ""
    chat_text_list = []

    for idx, row in messages_df.iterrows():
        if row["source"] == "A":
            attendant_message = f"A: {row['text']}"
            chat_text += attendant_message + "\n"
            chat_text_list.append(attendant_message)
        else:
            client_message = f"C: {row['text']}"
            chat_text += client_message + "\n"
            chat_text_list.append(client_message)

    return {"string": chat_text, "list": chat_text_list}


def format_client_messages_to_gpt(messages_df: pd.DataFrame):
    chat_text = ""
    chat_text_list = []

    for idx, row in messages_df.iterrows():
        if row["source"] == "C":
            client_message = f"C: {row['text']}"
            chat_text += client_message + "\n"
            chat_text_list.append(client_message)

    return {"string": chat_text, "list": chat_text_list}


# def create_report(formated_chat: list, sentiment_coef: float):
#     # Create byte buffer to hold report information
#     pdf_buffer = BytesIO()
#     # Create document
#     doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
#     # Report element list
#     elements = []
#     # Styles Class Instance
#     styles = getSampleStyleSheet()
#     # Creates a style for centered text
#     centered_style = PS(name="CenteredStyle", parent=styles["Heading3"], alignment=1)

#     # TITLE
#     title = Paragraph("Relatório de Análise de Sentimento do Chat", styles["Title"])
#     elements.append(title)
#     elements.append(Spacer(1, 20))

#     # SECTION - "Method for Sentiment Analysis"
#     section_title = Paragraph("Método para Análise de Sentimento", styles["Heading1"])
#     elements.append(section_title)

#     # SUBSECTION - "Method Description"
#     subtitle = Paragraph("Descrição do Método", styles["Heading2"])
#     elements.append(subtitle)
#     elements.append(Spacer(1, 10))

#     # TEXT - "method description"
#     method_introduction = (
#         "O método para a obtenção da estimativa do sentimento de um cliente durante "
#         "uma interação com o atendimento consiste em :"
#     )

#     text = Paragraph(method_introduction, styles["Normal"])
#     elements.append(text)
#     elements.append(Spacer(1, 10))

#     method_list = [
#         "Análise do sentimento de todas as mensagens dos clientes",
#         "Cálculo do peso de cada mensagem",
#         "Cálculo da média do sentimento do chat completo",
#         "Interpretação do resultado da análise",
#     ]

#     numbered_list = ListFlowable(
#         [
#             Paragraph(f"{item}", styles["Normal"])
#             for i, item in enumerate(method_list, start=1)
#         ],
#         bulletType="bullet",
#         leftIndent=20,
#     )
#     elements.append(numbered_list)
#     elements.append(Spacer(1, 10))

#     entry_1 = (
#         "O primeiro passo consiste na aplicação do modelo de aprendizado de máquina treinado para a "
#         "classificação do sentimento do cliente em cada uma das mensagens enviadas para o atendente, gerando assim "
#         'um nível estimado de satisfação do cliente que varia entre "Satisfeito", "Levemente Satisfeito", "Neutro"'
#         ', "Levemente Insatisfeito" ou "Insatisfeito".'
#     )

#     text = Paragraph(entry_1, styles["Normal"])
#     elements.append(text)
#     elements.append(Spacer(1, 10))

#     entry_2 = (
#         "O que se segue é a transformação das classificações dos sentimentos individuais "
#         "expressos em cada uma das mensagens em pesos matemáticos que compõem o sentimento do cliente "
#         "durante todo o atendimento. Esses pesos são definidos seguindo-se a metodologia formulada internamente"
#         " pelo time de Inteligência Artificial da ChatGuru."
#     )

#     text = Paragraph(entry_2, styles["Normal"])
#     elements.append(text)
#     elements.append(Spacer(1, 10))

#     entry_3 = (
#         "Usando-se parâmetros obtidos do chat completo e do modelo de IA da ChatGuru, é calculado um "
#         "coeficiente numérico de satisfação do atendimento completo."
#     )

#     text = Paragraph(entry_3, styles["Normal"])
#     elements.append(text)
#     elements.append(Spacer(1, 10))

#     entry_4 = (
#         "Por fim, esse coeficiente de satisfação é interpretado em termos não-matemáticos para ser "
#         "apreciado pelo contratante do serviço."
#     )

#     text = Paragraph(entry_4, styles["Normal"])
#     elements.append(text)
#     elements.append(Spacer(1, 20))
#     elements.append(PageBreak())

#     # SECTION - "Sentiment Analysis"
#     section_title = Paragraph("Análise de Sentimento", styles["Heading1"])
#     elements.append(section_title)

#     # SUBSECTION - "Chat Presentation"
#     subtitle = Paragraph("Apresentação do Chat", styles["Heading2"])
#     elements.append(subtitle)
#     elements.append(Spacer(1, 10))

#     # TEXT - "chat content"
#     for line in formated_chat:
#         chat = Paragraph(line, styles["Normal"])
#         elements.append(chat)
#         elements.append(Spacer(1, 5))

#     # SUBSECTION - "Analysis result"
#     subtitle = Paragraph("Resultados da Análise", styles["Heading2"])
#     elements.append(subtitle)
#     elements.append(Spacer(1, 10))

#     # TEXT - "analysis result intro"
#     analysis_result = (
#         "Ao se aplicar o método já descrito neste relatório, o coeficiente de satisfação do usuário na "
#         "conversa apresentada como objeto de análise foi de:"
#     )
#     text = Paragraph(analysis_result, styles["Normal"])
#     elements.append(text)
#     elements.append(Spacer(1, 8))

#     # TEXT - "sentiment coefficient"
#     str_coef = str(round(sentiment_coef, 3))
#     text = f"coeficiente de satisfação = {str_coef}"
#     centered_text = Paragraph(text, centered_style)
#     elements.append(centered_text)
#     elements.append(Spacer(1, 10))

#     # TEXT - "result interpretation"
#     sentiment_label = generate_sentiment_label(sentiment_coef)
#     interpretation = f"Dado o coeficiente de satisfação apresentado, podemos estimar que o cliente se sentiu:"
#     text = Paragraph(interpretation, styles["Normal"])
#     elements.append(text)

#     centered_text = Paragraph(sentiment_label, centered_style)
#     elements.append(centered_text)
#     elements.append(Spacer(1, 10))

#     # Build the rest of the report
#     doc.build(elements)

#     # Move buffer position to the beginning
#     pdf_buffer.seek(0)

#     return pdf_buffer


# # def update_file_to_s3(data, s3_bucket, s3_path):
# #     s3 = boto3.client("s3")
# #     try:
# #         s3.upload_fileobj(data, s3_bucket, s3_path)
# #     except Exception:
# #         raise S3UploadError(
# #             "Houve um erro ao fazer o upload do arquivo para o bucket S3"
# #         )
