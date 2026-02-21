INCLUDE(BuildPlugin)

BUILD_PLUGIN(aiaudiobridge
	AIAudioBridge.cpp
	AIAudioBridgeControls.cpp
	AIAudioBridgeControlDialog.cpp
	AIAudioBridgeTransport.cpp
	MOCFILES
		AIAudioBridgeControls.h
		AIAudioBridgeControlDialog.h
)
