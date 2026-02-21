/*
 * AIAudioBridge.h - realtime AI audio bridge effect
 */

#ifndef LMMS_AI_AUDIO_BRIDGE_H
#define LMMS_AI_AUDIO_BRIDGE_H

#include <vector>

#include "AIAudioBridgeControls.h"
#include "AIAudioBridgeTransport.h"
#include "Effect.h"

namespace lmms
{

class AIAudioBridgeEffect : public Effect
{
public:
	AIAudioBridgeEffect(Model* parent, const Descriptor::SubPluginFeatures::Key* key);
	~AIAudioBridgeEffect() override = default;

	ProcessStatus processImpl(SampleFrame* buf, const fpp_t frames) override;
	void processBypassedImpl() override;

	EffectControls* controls() override
	{
		return &m_controls;
	}

private:
	AIAudioBridgeControls m_controls;
	AIAudioBridgeTransport m_transport;
	std::vector<SampleFrame> m_inputScratch;
	std::vector<SampleFrame> m_processedScratch;
	AIAudioBridgeTransport::Config m_lastConfig;
	bool m_hasConfig = false;
};

} // namespace lmms

#endif // LMMS_AI_AUDIO_BRIDGE_H
