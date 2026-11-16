import re
s=open('supp.tex').read()
def inl(m):
    return open(m.group(1)+'.tex').read()
s2=re.sub(r'\\input\{(sec/supp/[a-z_0-9]+)\}', lambda m: open(m.group(1)+'.tex').read(), s)
open('supp_build.tex','w').write(s2)
print('inlined', len(re.findall(r'\\input\{sec/supp', s)))
