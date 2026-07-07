'''Streamlit dashboard for the atlas-registration probe-track plugin.
'''
import streamlit as st

from .common import schema_reference
from .sessions import _sessions_tab
from .track_annotation import _track_annotation_tab
from .alignment import _alignment_tab

dashboard_name = 'Atlas Registration'


def dashboard_function(schema=None):
    from ..pluginschema import (AtlasRegistration, AtlasRegistrationParams,
                                AtlasRegistrationAnnotation, ProbeTrack,
                                ProbeAlignment)
    st.write('## Atlas Registration — probe tracks')
    sessions_tab, track_tab, align_tab = st.tabs(
        ['Sessions', 'Track annotation', 'Alignment'])

    with sessions_tab:
        _sessions_tab(schema, AtlasRegistration, AtlasRegistrationParams,
                      AtlasRegistrationAnnotation, ProbeTrack)
        schema_reference('AtlasRegistration', 'AtlasRegistrationAnnotation',
                         'ProbeTrack')
    with track_tab:
        _track_annotation_tab(schema, AtlasRegistration, AtlasRegistrationParams,
                              AtlasRegistrationAnnotation, ProbeTrack)
        schema_reference('AtlasRegistrationAnnotation', 'ProbeTrack')
    with align_tab:
        _alignment_tab(schema, AtlasRegistration, AtlasRegistrationParams,
                       AtlasRegistrationAnnotation, ProbeTrack, ProbeAlignment)
        schema_reference('ProbeTrack', 'ProbeAlignment')
