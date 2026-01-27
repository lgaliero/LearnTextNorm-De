import re

ABBREV_VARIANTS = [
    r'w\.\s*z\.\s*[bB]\.?',   # w.z.B, w. z. B, w.z.B.
    r'[zZ]\.\s*[bB]\.?',       # z.B, z. B, z.B., Z.B
    r'[zZ][bB]\.?',            # zB, ZB, zB., ZB.
    r'u\.s\.w\.?',             # u.s.w, u.s.w.
    r'u\.n\.w\.?',             # u.n.w.
    r'u\.a\.?',                # u.a, u.a.
    r'd\.h\.?',                # d.h, d.h.
    r'c\.a\.?',                # c.a, c.a.
    r'o\.\s*[äÄ]\.?',          # o.ä, o. Ä, o.ä.
    r'o\.',                    # o. (standalone)
    r'[oO]\.[kK]\.?',          # o.k., O.K., o.k
    r'[U]\.?[A]\.?',            # U.A, U.A. etc.
    r'M\.S\.?',                # M.S, M.S.
    r'[A-ZÄÖÜ]\.',             # Single capital letter abbreviations: H., P., M., etc.
    r'Min\.', r'min\.', r'bzw\.', r'usw\.', r'etc\.', r'ecc\.', r'ca\.', r'evtl\.', 
    r'ggf\.', r'inkl\.', r'max\.', r'Nr\.', r'Tel\.', r'vs\.', 
    r'Mr\.', r'Mrs\.', r'Ms\.', r'Dr\.', r'Prof\.', r'Fam\.'
]


# Keep original ABBREVIATIONS for sentencizer
ABBREVIATIONS = [
    r'bo\.\s',
    r'o\.\s?ä',
    r'o\.\sÄ',
    r'[zZ]\.\s?[bB]',
    r'\bw\.\s*z\.\s*[bB]\.?\b',
    r'\bu\.?s\.?w\.?\)?\b',
    r'u\.n\.w',
    r'u\.a',
    r'd\.h',
    r'c\.a',
    r'\b[oO]\.?[kK]\.?',
    r'P\.S',
    r'Min', r'min', r'bzw', r'usw', r'etc', r'ecc', r'ca', r'evtl', 
    r'ggf', r'inkl', r'max', r'Nr', r'Tel', r'vs', r'Mr', r'Mrs', 
    r'Ms', r'Dr', r'Prof', r'Fam'
]

# Compile pattern to detect any abbreviation with optional spacing
ABBREV_PATTERN = re.compile(
    r'\b(' + '|'.join(ABBREVIATIONS) + r')\.?\b',
    re.IGNORECASE
)

# All possible quote characters
QUOTE_CHARS = {'"', '„', '"', '"', '«', '»', '‹', '›'}
