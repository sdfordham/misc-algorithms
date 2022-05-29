DIGITS = '0123456789'
MATHOPS = '-+/x^'
BRACKETS = '()'


def tokenize(s: str) -> list[str | int]:
    s = s.replace(' ', '')
    idx, n = 0, len(s)
    toks = list()
    while(idx < n):
        if s[idx] in DIGITS:
            tok = ''
            while idx < n and s[idx] in DIGITS:
                tok += s[idx]
                idx += 1
            toks.append(int(tok))
        elif s[idx] in MATHOPS or s[idx] in BRACKETS:
            toks.append(s[idx])
            idx += 1
        else:
            raise ValueError('Something unexpected:', s[idx])
    return toks


def rpn(exp: str) -> float:
    """
    Reverse Polish notation
    """
    toks = tokenize(exp)[::-1]
    oper_stack, out_que = list(), list()
    while toks:
        tok = toks.pop()
        if isinstance(tok, int):
            out_que.append(tok)
        elif tok in MATHOPS:
            prio = MATHOPS.index(tok)
            if oper_stack and oper_stack[-1] in MATHOPS:
                while oper_stack and prio < MATHOPS.index(oper_stack[-1]):
                    out_que.append(oper_stack.pop())
            oper_stack.append(tok)
        elif tok in BRACKETS:
            if tok == '(':
                oper_stack.append(tok)
            else:
                while oper_stack and oper_stack[-1] != '(':
                    out_que.append(oper_stack.pop())
                if not oper_stack or oper_stack[-1] != '(':
                    raise ValueError('Bad exp, closing bracket missing.')
                else:
                    oper_stack.pop()
    while oper_stack:
        out_que.append(oper_stack.pop())
    return out_que


if __name__ == '__main__':
    s = '4 + 18 / (9 - 3)'
    print(s)
    print(rpn(s))

    s = '(7 - 3) x 5 / 8'
    print(s)
    print(rpn(s))