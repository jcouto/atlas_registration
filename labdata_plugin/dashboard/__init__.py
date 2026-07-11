'''Streamlit dashboard for the atlas-registration and probe-tracing plugin.
'''

dashboard_name = 'Atlas Registration'


def dashboard_function(schema=None):
    import streamlit as st
    from .common import schema_reference
    from .sessions import _sessions_tab
    from .track_annotation import _track_annotation_tab
    from .alignment import _alignment_tab

    from ..pluginschema import (AtlasRegistration, AtlasRegistrationParams,
                                AtlasRegistrationAnnotation, ProbeTrack,
                                ProbeAlignment)
    st.write('## Atlas Registration — probe tracks')

    # using segmented control so only the active tab's body runs each rerun 
    tabs = ['Sessions', 'Track annotation', 'Alignment']
    active = st.segmented_control('view', tabs, default='Sessions',
                                  selection_mode='single', key='ar_active_tab',
                                  label_visibility='collapsed')
    active = active or 'Sessions'   # segmented_control can return None

    if active == 'Sessions':
        _sessions_tab(schema, AtlasRegistration, AtlasRegistrationParams,
                      AtlasRegistrationAnnotation, ProbeTrack)
        schema_reference('AtlasRegistration', 'AtlasRegistrationAnnotation',
                         'ProbeTrack')
    elif active == 'Track annotation':
        _track_annotation_tab(schema, AtlasRegistration, AtlasRegistrationParams,
                              AtlasRegistrationAnnotation, ProbeTrack)
        schema_reference('AtlasRegistrationAnnotation', 'ProbeTrack')
    elif active == 'Alignment':
        _alignment_tab(schema, AtlasRegistration, AtlasRegistrationParams,
                       AtlasRegistrationAnnotation, ProbeTrack, ProbeAlignment)
        schema_reference('ProbeTrack', 'ProbeAlignment')
