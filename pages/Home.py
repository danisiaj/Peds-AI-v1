import streamlit as st

def log_in():
    """
    This function builds a layer of security for log in feature

    Return: logged_in status, bool
    """
    # Check login state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Display login form if not logged in
    if not st.session_state.logged_in:
        with st.sidebar.form(key="login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            # Submit button within the form
            login_button = st.form_submit_button("Login")
            
            if login_button:
                # Check username and password
                if username == "danisiaj" and password == "1234":
                    st.session_state.logged_in = True
                    st.session_state['username'] = username
                    st.sidebar.success("Login successful!")
                    return True
                else:
                    st.sidebar.error("Your login info is incorrect.")
        return False
    return True

def set_up_log_out_button():
    """
    This function sets up the log out button to add the option to end session
    """

    # Display logout button only if logged in
    if st.session_state.logged_in:
        if st.sidebar.button('Log out'):
            # Reset login state on logout
            st.session_state.logged_in = False
            st.rerun(scope='app')  # Refresh the app to show the login form again

def page_setup():
    """
    Set up basic page configuration
    """
    page_config = st.set_page_config(
        page_title="My Peds AI",
        page_icon=':baby:',
        menu_items={"Get Help": 'https://www.google.com'},
        layout='wide'

    )

    st.title('Pediatric AI Nursing Assistant')
    st.header('👶🏻 👶🏼 👶🏽 👶🏾 👶🏿')


    return page_config

def set_up_nav():
    """
    Initializes the different app pages and builds a navigation menu
    """

    pages = {
            "" : [
                st.Page("home.py", title='Home', icon='🏠')
                ],
            "Resources": [
                st.Page("Education.py", title="Education Center", icon='📚'),
                st.Page("peds_ai.py", title="Peds AI", icon='👶🏽'),
                st.Page("wound_classifier.py", title= "Wound Classifier", icon='🩹')
                ]
    }
      
    pg = st.navigation(pages, position="sidebar")
    pg.run()

def main():
    page_setup()
    # Check if user is logged in
    if log_in():
        set_up_nav()    # Show navigation if logged in
        set_up_log_out_button()  # Show logout button if logged in
    else:
        st.sidebar.info("Please log in to access the navigation options.")


if __name__ == "__main__":
    main()



