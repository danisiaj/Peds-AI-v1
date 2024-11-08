import streamlit as st
import pandas as pd
import sqlite3

def set_up_page():
    st.title("Education")
    st.write("Explore educational resources and guides here.")

def load_data():
    nurses_data = pd.read_csv('data/nurses_dataset.csv')

    return nurses_data

def sql_data(sql_query):
    # Connect to an in-memory SQLite database 
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Load and execute SQL file
    with open('data/nurses_1.sql', 'r') as file:
        sql_script = file.read()
    cursor.executescript(sql_script)  # Executes all commands in the SQL file
    conn.commit()

    df_sql = pd.read_sql_query(sql_query, conn)  # Use pandas to read SQL query results into a DataFrame

    # Display in Streamlit
    st.dataframe(df_sql, use_container_width=False)
# Close the database connection
    conn.close()

def sql_query():

    # Example query: Fetch all nurses with BLS certification expiring 
    sql_query = """SELECT 
                    _id, 
                    bls_expiration_countown 
                    FROM education_info 
                    WHERE bls_expiration_countown > 0
                    ORDER BY bls_expiration_countown asc"""
    
    return sql_query

def main():

        
    set_up_page()
    nurses_data = load_data()
    st.markdown('### _My Nurses:_')
    st.dataframe(nurses_data.head(), use_container_width=True, hide_index=True)

    st.markdown('### _BLS:_')
    sqlquery = sql_query()
    sql_data(sqlquery)

    st.markdown('### _Query History:_')
    # Initialize 'nurses_data' in session state if it doesn't exist
    if 'nurses_data' not in st.session_state:
        st.session_state['nurses_data'] = pd.DataFrame(columns=['username', 'question'])
        
    if st.session_state.get('username') is not None and st.session_state.get('query') is not None:
    # Create a new row as a dictionary
        new_row = pd.DataFrame([{'username': st.session_state['username'], 'question': st.session_state['query']}])
        
        # Append the new row to the nurses_data dataframe
        st.session_state['nurses_data'] = pd.concat([st.session_state['nurses_data'], new_row], ignore_index=True)
        
        # Display the updated dataframe
        st.dataframe(st.session_state['nurses_data'], use_container_width=True)

main()