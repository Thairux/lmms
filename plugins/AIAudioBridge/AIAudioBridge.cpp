/*
 * AIAudioBridge.cpp - realtime AI audio bridge effect
 */

#include "AIAudioBridge.h"

#include <cmath>

#include "plugin_export.h"

namespace lmms
{

extern "C"
{

Plugin::Descriptor PLUGIN_EXPORT aiaudiobridge_plugin_descriptor =
{
	LMMS_STRINGIFY(PLUGIN_NAME),
	"AI Audio Bridge",
	QT_TRANSLATE_NOOP("PluginBrowser",
		"Realtime AI bridge with pipe/shared-memory transport and local/cloud backends"),
	"LMMS community",
	0x0100,
	Plugin::Type::Effect,
	nullptr,
	nullptr,
	nullptr,
} ;

PLUGIN_EXPORT Plugin* lmms_plugin_main(Model* parent, void* data)
{
	return new AIAudioBridgeEffect(parent, static_cast<const Plugin::Descriptor::SubPluginFeatures::Key*>(data));
}

}


AIAudioBridgeEffect::AIAudioBridgeEffect(Model* parent, const Descriptor::SubPluginFeatures::Key* key) :
	Effect(&aiaudiobridge_plugin_descriptor, parent, key),
	m_controls(this)
{
}


Effect::ProcessStatus AIAudioBridgeEffect::processImpl(SampleFrame* buf, const fpp_t frames)
{
	if (!m_controls.m_bridgeEnabledModel.value() || frames == 0)
	{
		return ProcessStatus::ContinueIfNotQuiet;
	}

	if (m_inputScratch.size() != frames)
	{
		m_inputScratch.resize(frames);
		m_processedScratch.resize(frames);
	}
	for (fpp_t i = 0; i < frames; ++i)
	{
		m_inputScratch[i] = buf[i];
	}

	const auto cfg = m_controls.transportConfig();
	if (!m_hasConfig ||
		cfg.transportMode != m_lastConfig.transportMode ||
		cfg.backend != m_lastConfig.backend ||
		cfg.command != m_lastConfig.command ||
		cfg.scriptPath != m_lastConfig.scriptPath ||
		cfg.apiKey != m_lastConfig.apiKey ||
		cfg.sampleRate != m_lastConfig.sampleRate ||
		cfg.channels != m_lastConfig.channels ||
		cfg.timeoutMs != m_lastConfig.timeoutMs ||
		std::abs(cfg.drive - m_lastConfig.drive) > 0.0001f)
	{
		m_transport.updateConfig(cfg);
		m_lastConfig = cfg;
		m_hasConfig = true;
	}

	m_transport.submit(m_inputScratch.data(), frames);
	const bool hasProcessed = m_transport.consume(m_processedScratch.data(), frames);

	const float aiMix = m_controls.m_mixModel.value() * 0.01f;
	const float d = dryLevel();
	const float w = wetLevel();
	for (fpp_t i = 0; i < frames; ++i)
	{
		const auto processed = hasProcessed ? m_processedScratch[i] : m_inputScratch[i];
		const auto aiBlend = m_inputScratch[i] * (1.0f - aiMix) + processed * aiMix;
		buf[i] = m_inputScratch[i] * d + aiBlend * w;
	}

	return ProcessStatus::ContinueIfNotQuiet;
}


void AIAudioBridgeEffect::processBypassedImpl()
{
}

} // namespace lmms
