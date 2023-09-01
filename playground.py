import streamlit as st
import plotly.express as px
import pandas as pd
import json
import numpy as np

def shifted_log1p(x):
    shift = np.abs(np.min(x)) + 1
    return np.log1p(x + shift)

st.set_page_config(page_title='umlcaxs playground', page_icon = "🔭", layout='wide')
# Centering the title using markdown

st.markdown("<h1 style='text-align: center;'>Unsupervised Machine Learning for the Classification of Astrophysical X-ray Sources</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Uniquely classified sources playground</h5>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center">
    Read the paper <a href="#">here</a> (soon).
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center">
    <a href="https://github.com/samuelperezdi/umlcaxs_playground">source</a>
</div>
""", unsafe_allow_html=True)

st.markdown("""
---
<div style="text-align: center">
    Víctor Samuel Pérez-Díaz<sup>1,2</sup><a href="mailto:samuelperez.di@gmail.com">✉</a>, Juan Rafael Martínez-Galarza<sup>1</sup>, Alexander Caicedo<sup>3, 4</sup>, and Raffaele D'Abrusco<sup>1</sup>  
</div>
<br>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center"> 
    <sup>1</sup>Center for Astrophysics | Harvard & Smithsonian, 60 Garden Street, Cambridge, MA 02138, USA <br>
    <sup>2</sup>School of Engineering, Science and Technology, Universidad del Rosario, Cll. 12C No. 6-25, Bogotá, Colombia <br> 
    <sup>3</sup>Department of Electronics Engineering, Pontificia Universidad Javeriana, Cra. 7 No. 40-62, Bogotá, Colombia <br>
    <sup>4</sup>Ressolve, Cra. 42 # 5 Sur - 145, Medellín, Colombia 
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Read data
df_class_confident_with_coords = pd.read_csv('./out_data/uniquely_classified.csv', encoding='utf-8')

# Dropdown to select visualization type
vis_type = st.selectbox("Visualization", ["Aladin Sky Atlas", "Properties"])

with st.expander("Documentation"):
    st.write("""
This web app offers two distinct visualization options for exploring our uniquely classified table of sources. Each option provides interactive features to help you engage with the data in a meaningful way.
             
**Aladin Sky Atlas:**
The Aladin Sky Atlas visualization displays each astronomical source, color-coded by its aggregated class.

- Class Selection: To filter which classes are displayed, click on the stack icon located in the top-left corner of the visualization.

- Object Search: If you are interested in particular celestial objects, you can search for them by clicking on the magnifying glass (Look) icon.

For additional information and functionalities on the Aladin Lite tool, consult the [official documentation](http://aladin.cds.unistra.fr/AladinLite/doc/).

**Properties:** 
This scatter plot visualization enables an in-depth look at various properties used in the data pipeline.

- Class and Feature Selection: Use the interactive controls to select which classes to display and to choose the features plotted on the X and Y axes.

- Color Dimension: While the default color scheme represents the aggregated class, you have the option to enable a different property to be depicted in color.

- Log Transformation: For more nuanced exploration, a log transformation can be applied to any of the plotted dimensions.

This tool provides a flexible and informative way to interact with and analyze the characteristics of the classified sources. The data used in this web app, along with additional datasets and code, are available in the [paper's GitHub repository](https://github.com/samuelperezdi/umlcaxs). For a comprehensive understanding of the pipeline and methodologies employed, please refer to our [paper](#).
    """, unsafe_allow_html=True)

if vis_type == "Aladin Sky Atlas":
# Extract sources and add a 'type' column (assuming you have this data)

    # Multiselect widget to let users select which classes to visualize
    #classes_to_show = st.multiselect('Aggregated classes to visualize', ['AGN', 'Seyfert', 'YSO', 'XB'], default=['AGN', 'Seyfert', 'YSO', 'XB'])

    # Filter the dataframe based on the selected classes
    #df_class_confident_filtered = df_class_confident_with_coords[df_class_confident_with_coords['agg_master_class'].isin(classes_to_show)]

    data_to_serialize = df_class_confident_with_coords.to_dict(orient='records')

    # Load the Aladin Lite HTML and replace the placeholder
    with open('aladin_lite.html', 'r') as f:
        html_code = f.read().replace('DATA_PLACEHOLDER', json.dumps(data_to_serialize))

    st.components.v1.html(html_code, height=420)

elif vis_type == "Properties":
    features = [
        'hard_hm', 'hard_hs', 'hard_ms', 'powlaw_gamma', 'bb_kt', 
        'var_prob_b','var_ratio_b', 'var_prob_h', 'var_ratio_h', 
        'var_prob_s', 'var_ratio_s', 'var_newq_b'
    ]

    # Class selection
    classes_to_show = st.multiselect('Aggregated classes to visualize', 
                                ['AGN', 'Seyfert', 'YSO', 'XB'], 
                                default=['AGN', 'Seyfert', 'YSO', 'XB'])

    if not classes_to_show:
        st.warning("Please select at least one class.")
        st.stop()
    df_filtered = df_class_confident_with_coords[df_class_confident_with_coords['agg_master_class'].isin(classes_to_show)]
    
    col1, col2, col3 = st.columns(3)
    # Feature selection dropdowns
    feature1 = col1.selectbox('Select feature for x-axis', features, index=features.index('hard_hm'))
    feature2 = col2.selectbox('Select feature for y-axis', features, index=features.index('hard_hs'))

    # Log transformation checkboxes
    log_feature1 = col1.checkbox(f'Log transform x-axis')
    log_feature2 = col2.checkbox(f'Log transform y-axis')

    # Color dimension activation
    color_option = col3.checkbox('Activate color dimension')
    color_feature = None
    if color_option:
        color_feature = col3.selectbox('Select feature for color dimension', features)
        log_color_feature = col3.checkbox(f'Log transform color')

    # Plot
    color = color_feature if color_option else 'agg_master_class'
    type_colors = None if color_option else {
                    'AGN': '#1f77b4',      # matplotlib's default blue
                    'Seyfert': '#ff7f0e',  # orange
                    'XB': '#2ca02c',       # matplotlib's default green
                    'YSO': '#d62728'       # matplotlib's default red
                }


    # Log transformations
    df_plot = df_filtered.copy()

    if log_feature1:
        df_plot[f"log({feature1})"] = shifted_log1p(df_filtered[feature1])
        x_column = f"log({feature1})"
    else:
        x_column = feature1

    if log_feature2:
        df_plot[f"log({feature2})"] = shifted_log1p(df_filtered[feature2])
        y_column = f"log({feature2})"
    else:
        y_column = feature2

    x_title = f"log({feature1})" if log_feature1 else feature1
    y_title = f"log({feature2})" if log_feature2 else feature2

    # Log transformation for color
    if color_option:
        color_title = f"log({color_feature})" if log_color_feature else color_feature
        if log_color_feature:
            df_plot[f"log({color_feature})"] = shifted_log1p(df_filtered[color_feature])
            color_column = f"log({color_feature})"
        else:
            color_column = color_feature
    else:
        color_column = 'agg_master_class'
        color_title = 'Class'

    fig = px.scatter(df_plot,
                    x=x_column,
                    y=y_column,
                    color=color_column,
                    color_discrete_map=type_colors if not color_option else None,
                    hover_name='name',
                    hover_data=[feature1, feature2, color_feature],
                    color_continuous_scale="inferno_r")

    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        coloraxis_colorbar_title=color_title
    )

    if log_feature1:
        fig.update_xaxes(type="log")
    if log_feature2:
        fig.update_yaxes(type="log")
    #if color_option and log_color_feature:
        #fig.update_coloraxes(dtick="log")

    st.plotly_chart(fig, use_container_width=True, use_container_height=True)

