from dash import Dash, html


app = Dash(__name__)
app.layout = html.Div((html.H1("Dash is running!"), html.P("Content goes here.")))

if __name__ == "__main__":
    app.run(debug=True)
