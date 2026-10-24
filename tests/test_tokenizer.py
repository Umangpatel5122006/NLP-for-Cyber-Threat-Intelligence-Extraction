from cti.preprocessing.tokenizer import tokenize

text = "APT29 used Cobalt Strike, then contacted evil.example.com (185.22.10.5)."
for tok in tokenize(text):
    print(tok)

print(tokenize("यह हमला बहुत खतरनाक है।", lang="hi"))