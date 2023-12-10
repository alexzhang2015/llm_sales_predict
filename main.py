from openai import OpenAI
import gradio as gr
import pandas as pd
import pickle
from dotenv import dotenv_values

env = dotenv_values()
client = OpenAI(api_key=env['OPENAI_API_KEY'], base_url=env['OPENAI_API_BASE'])

llm_model = "gpt-4-1106-preview"

def run_predict_sales(content):
    print(f"the content is as follows: {content}..")
    # Step 1: send the conversation and available functions to the model
    messages = [
        {"role": "user", "content": f"You are a great business analysis expert. You call the predict_sales function to conduct sales forecasts. You mainly extract the following user input and extract sales data and days data, the content is as follows: {content}. Please express the result in Chinese."}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "predict_sales",
                "description": "predict sales by sales prediction model",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sales": {
                            "type": "number",
                            "description": "The sales for some given day, e.g. 38033.4",
                        },
                        "day": {
                            "type": "string",
                            "description": "Some given day, e.g. 2023-01-09",
                        },
                    },
                    "required": ["sales"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model=llm_model,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "predict_sales": predict_sales,
        }  # only one function in this example, but you can have multiple
        # extend conversation with assistant's reply
        messages.append(response_message)
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                sales=function_args.get("sales"),
                day=function_args.get("day"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model=llm_model,
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response.choices[0].message.content


def predict_sales(sales, day='2023-01-09'):
    # load model from file
    # This file should be in the current working derectory
    pkl_filename = 'PredictionModel/StoreSalesPrediction.pkl'

    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    # predict sales for a future date. Here is using a fix date '2023-01-09' because the history data is only up to '2023-01-08' and this model is base on the last day history sales data.
    X_test = pd.DataFrame({
        'Lag_1': [sales]
    }, index=[day])

    y_pred_test = pd.Series(pickle_model.predict(X_test), index=X_test.index)

    return str(y_pred_test)


with gr.Blocks() as demo:
    gr.Markdown(
        "## Start typing below and then click **Run** to see the output.")
    with gr.Row():
        inp = gr.Textbox(placeholder="City Weather?")
        out = gr.Textbox()
    btn = gr.Button("Run")
    btn.click(fn=run_predict_sales, inputs=inp, outputs=out)

# 添加按钮到界面
demo.launch(share=True)
# demo.launch(share=True, auth=("username", "password"))
