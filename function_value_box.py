import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Define a function to create value boxes
def create_value_box(value, color, caption, icon):
    value_box = go.Box(
        name=caption,
        x=[value],
        marker_color=color,
        line_color="black",
        text=icon,
        boxpoints=False,
        jitter=0,
        pointpos=0,
        boxmean=True
    )
    return value_box

# Create the value boxes
value_box_36_months = create_value_box(
    10.5, "blue", "36-Month Loan (Last Quarter Avg Interest Rate)", "👍"
)

value_box_60_months = create_value_box(
    12.0, "green", "60-Month Loan (Last Quarter Avg Interest Rate)", "👍"
)

# Create subplots
fig = make_subplots(rows=1, cols=2)

# Add value boxes to the subplots
fig.add_trace(value_box_36_months, row=1, col=1)
fig.add_trace(value_box_60_months, row=1, col=2)

# Customize subplot layout
fig.update_layout(
    template="plotly_white",  # You can change the template as needed
    title="Your Dashboard Title",
    height=400,  # Set the height of the entire dashboard
    margin=dict(l=20, r=20, t=50, b=20),  # Adjust margins as needed
)

# Show the dashboard
fig.show()
