import streamlit_authenticator as stauth

passwords = ['reine', 'suisei']
hash_pass = stauth.Hasher(passwords).generate()

print(hash_pass)