"""Official KINNEWS category order (Niyongabo et al., 2020)."""
LABELS = ('politics','sport','economy','health','entertainment','history','technology','tourism','culture','fashion','religion','environment','education','relationship')
KIN = ('Politiki','Imikino','Ubukungu','Ubuzima','Imyidagaduro','Amateka','Ikoranabuhanga','Ubukerarugendo','Umuco','Imideli','Iyobokamana','Ibidukikije','Uburezi','Imibanire')

def name(value):
    try:
        index = int(value)
        return LABELS[index] if 0 <= index < len(LABELS) else str(value)
    except (ValueError,TypeError,IndexError):
        return str(value)

def kin_name(value):
    english = name(value)
    return KIN[LABELS.index(english)] if english in LABELS else english
