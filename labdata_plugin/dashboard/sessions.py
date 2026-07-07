import pandas as pd
import streamlit as st

from .common import tab_cache_factory, refresh_button, _ANNOTATED_COLOR


def _sessions_tab(schema, AtlasRegistration, AtlasRegistrationParams,
                  AtlasRegistrationAnnotation, ProbeTrack):
    cache = tab_cache_factory('refresh_ar_sessions')
    refresh_button('refresh_ar_sessions')

    @cache
    def get_counts():
        out = {}
        for name, tbl in [('Registrations', AtlasRegistration()),
                          ('Annotations', AtlasRegistrationAnnotation()),
                          ('ProbeTracks', ProbeTrack())]:
            try:
                out[name] = len(tbl)
            except Exception:
                out[name] = None
        return out

    counts = get_counts()
    cols = st.columns(len(counts))
    for col, (name, val) in zip(cols, counts.items()):
        col.metric(name, '—' if val is None else val)

    @cache
    def get_registrations():
        rows = (AtlasRegistration * AtlasRegistrationParams).proj(
            'atlas', 'brain_geometry').fetch(as_dict=True)
        return pd.DataFrame(rows)

    regs = get_registrations()
    if not regs.empty:
        # counts are computed fresh each run so fitting a track elsewhere shows up.
        # Restrict only by the registration primary key, coercing numpy scalars to
        # native Python types (datajoint restrictions choke on numpy ints).
        key_attrs = [c for c in AtlasRegistration().primary_key if c in regs.columns]

        def _row_key(r):
            return {c: (r[c].item() if hasattr(r[c], 'item') else r[c])
                    for c in key_attrs}

        regs = regs.copy()
        regs['n_annotations'] = [len(AtlasRegistrationAnnotation & _row_key(r))
                                 for _, r in regs.iterrows()]
        regs['n_tracks'] = [len(ProbeTrack & _row_key(r))
                            for _, r in regs.iterrows()]
    if regs.empty:
        st.info('No AtlasRegistration rows found. Populate AtlasRegistration first.')
        return

    if 'subject_name' in regs.columns:
        subjects = sorted(regs['subject_name'].unique())
        subject = st.selectbox('Subject', subjects, index=None, key='ar_subject')
        if subject:
            regs = regs[regs['subject_name'] == subject]

    def _highlight(row):
        color = _ANNOTATED_COLOR if row.get('n_tracks', 0) > 0 else ''
        return [f'background-color: {color}'] * len(row)

    event = st.dataframe(regs.style.apply(_highlight, axis=1), hide_index=True,
                         width='stretch', on_select='rerun',
                         selection_mode='single-row', key='ar_reg_table')
    sel = (event.selection or {}).get('rows', [])
    if not sel or sel[0] >= len(regs):
        st.caption('Click a registration row to select it. Green rows already '
                   'have fitted ProbeTracks.')
        return

    row = regs.iloc[sel[0]]
    pk = [c for c in regs.columns
          if c not in ('atlas', 'brain_geometry', 'n_annotations', 'n_tracks')]
    sel_key = {c: row[c] for c in pk}
    st.session_state['ar_selected_key'] = sel_key
    st.session_state['ar_selected_atlas'] = row['atlas']
    st.session_state['ar_selected_geometry'] = row['brain_geometry']

    st.divider()
    st.write(f"**Selected:** `{sel_key}` — atlas **{row['atlas']}** "
             f"({row['brain_geometry']}), {int(row['n_annotations'])} shank "
             f"annotation(s), {int(row['n_tracks'])} track(s).")
    st.caption('Go to the **Track annotation** tab to trace/fit tracks, then '
               '**Alignment** to align channels, and **Read-out** for the tables.')
